"""
dit_backbone.py — DiT 风格速度场网络
=====================================
现代化 Diffusion Transformer (DiT) 架构，用于 Flow Matching 速度场预测。

相比朴素 Transformer 的关键改进:

1. **Patchify Tokenization**
   将 data_dim 切分为 num_patches 个 token，让 Self-Attention 真正有序列可建模。
   旧方案只有 3 个 token ([x_t, t, cond])，注意力几乎退化为加权求和。

2. **AdaLN-Zero 条件调制** (DiT, Peebles & Xie 2023)
   时间 + 条件 → 每个 Block 的 (shift, scale, gate) × 2。
   gate 零初始化 → 训练初期每个 Block 近似恒等映射，深层网络也能稳定训练。

3. **QK-Norm** (Dehghani et al. 2023)
   Query/Key 做 RMSNorm 后再算注意力，抑制 attention logits 爆炸，
   是大规模模型训练稳定性的关键技术。

4. **RoPE (旋转位置编码)**
   给 patch token 注入相对位置信息，外推性优于绝对 position embedding。

5. **SwiGLU FFN**
   FC_down(Swish(FC_gate(x)) ⊙ FC_up(x))，表达力优于标准 GELU FFN。

6. **RMSNorm**
   无 bias、不计算 mean，比 LayerNorm 更轻更稳。

7. **Gradient Checkpointing**
   大规模训练时用计算换显存。

8. **Zero-init 输出层**
   最后一层零初始化 → 训练初期输出零速度场，等价于恒等流。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ────────────────────────────────────────────────────────────
# 规模预设
# ────────────────────────────────────────────────────────────

DIT_PRESETS = {
    # preset: (dim, depth, heads, num_patches)
    "tiny":  (384,  4,  6,  8),    # ~10M   快速调试
    "small": (768,  8,  12, 16),   # ~80M   小数据集
    "base":  (1024, 16, 16, 16),   # ~250M  推荐默认
    "large": (1536, 32, 16, 16),   # ~1.2B  大数据集
    "xl":    (2048, 48, 16, 32),   # ~4B    超大规模
}


def resolve_dit_preset(preset: str) -> dict:
    """将 preset 名称解析为具体超参数"""
    if preset not in DIT_PRESETS:
        raise ValueError(
            f"Unknown dit_preset: {preset}, choose from {list(DIT_PRESETS.keys())}"
        )
    dim, depth, heads, num_patches = DIT_PRESETS[preset]
    return {
        "dit_dim": dim,
        "dit_depth": depth,
        "dit_heads": heads,
        "dit_num_patches": num_patches,
    }


# ────────────────────────────────────────────────────────────
# 基础组件
# ────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square LayerNorm。

    相比 LayerNorm: 不计算 mean、无 bias，计算更省且训练更稳。
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 用 float32 计算 rsqrt 保证数值稳定 (AMP 场景下重要)
        dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = (x_fp32 * rms).to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间编码"""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=device, dtype=torch.float32)
            / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class SwiGLU(nn.Module):
    """
    SwiGLU FFN: FC_down(Swish(FC_gate(x)) ⊙ FC_up(x))

    参数量约为 3 * dim * hidden，所以 hidden 通常取 mlp_ratio * dim * 2/3
    以对齐标准 FFN (2 * dim * mlp_ratio * dim) 的参数量。
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.fc_up = nn.Linear(dim, hidden_dim, bias=False)
        self.fc_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_down(F.silu(self.fc_gate(x)) * self.fc_up(x)))


# ────────────────────────────────────────────────────────────
# RoPE (Rotary Position Embedding)
# ────────────────────────────────────────────────────────────

def build_rope_cache(seq_len: int, head_dim: int, device, base: float = 10000.0):
    """
    预计算 RoPE 的 cos/sin 表。

    Returns:
        cos, sin: 各 (seq_len, head_dim)
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)            # (seq_len, half)
    emb = torch.cat([freqs, freqs], dim=-1)       # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    对 Q/K 应用旋转位置编码。

    Args:
        x: (batch, heads, seq_len, head_dim)
        cos, sin: (seq_len, head_dim)
    """
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotated * sin


# ────────────────────────────────────────────────────────────
# Attention with QK-Norm + RoPE
# ────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """
    Multi-Head Self-Attention with QK-Norm + RoPE。

    QK-Norm: 对 Q/K 分别做 RMSNorm 再算注意力，抑制 logits 爆炸。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qk_norm: bool = True,
        use_rope: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) 必须能被 num_heads({num_heads}) 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_rope = use_rope

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout_p = dropout

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # RoPE cache (lazy 构建，随 seq_len / device 变化重建)
        self._rope_cache = None
        self._rope_meta = None

    def _get_rope(self, seq_len: int, device):
        meta = (seq_len, self.head_dim, device)
        if self._rope_meta != meta:
            self._rope_cache = build_rope_cache(seq_len, self.head_dim, device)
            self._rope_meta = meta
        return self._rope_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.use_rope:
            cos, sin = self._get_rope(T, x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


# ────────────────────────────────────────────────────────────
# DiT Block with AdaLN-Zero
# ────────────────────────────────────────────────────────────

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN 调制: x * (1 + scale) + shift, shift/scale 广播到 token 维"""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    DiT Block with AdaLN-Zero 条件调制。

    结构:
        h = x + gate_attn * Attention(modulate(RMSNorm(x), shift_1, scale_1))
        y = h + gate_ffn  * SwiGLU(   modulate(RMSNorm(h), shift_2, scale_2))

    其中 (shift_1, scale_1, gate_attn, shift_2, scale_2, gate_ffn)
    由条件向量 c 经过一个零初始化的线性层产生。
    零初始化 → 训练初期 gate=0 → Block 退化为恒等映射 → 深层网络稳定收敛。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        use_rope: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.attn = Attention(
            dim, num_heads, qk_norm=qk_norm, use_rope=use_rope, dropout=dropout
        )
        self.norm2 = RMSNorm(dim, elementwise_affine=False)

        # SwiGLU 有 3 个投影矩阵，取 2/3 保持与标准 FFN 参数量相当
        ffn_hidden = int(dim * mlp_ratio * 2 / 3)
        ffn_hidden = (ffn_hidden + 63) // 64 * 64      # 对齐到 64 的倍数
        self.ffn = SwiGLU(dim, ffn_hidden, dropout=dropout)

        # AdaLN-Zero: 条件 → 6 组调制参数
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        # 零初始化 → 初期恒等映射
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_ln(c).chunk(6, dim=-1)

        x = x + gate1.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2.unsqueeze(1) * self.ffn(modulate(self.norm2(x), shift2, scale2))
        return x


class FinalLayer(nn.Module):
    """
    输出层: AdaLN 调制 → Linear (零初始化)。

    零初始化保证训练初期预测零速度场，等价于恒等流，避免初期梯度震荡。
    """

    def __init__(self, dim: int, patch_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim, elementwise_affine=False)
        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True),
        )
        self.proj = nn.Linear(dim, patch_dim, bias=True)

        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(c).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


# ────────────────────────────────────────────────────────────
# DiT Velocity Network
# ────────────────────────────────────────────────────────────

class DiTVelocityNetwork(nn.Module):
    """
    DiT 风格速度场预测网络 v_θ(x_t, t, condition)。

    数据流:
        x_t (B, data_dim)
          │  Patchify: reshape → (B, num_patches, patch_dim) → Linear → (B, P, dim)
          ▼
        tokens (B, P, dim)
          │  + N × DiTBlock(tokens, c)      c = time_emb + cond_emb
          ▼
        tokens (B, P, dim)
          │  FinalLayer → (B, P, patch_dim) → Unpatchify
          ▼
        v_pred (B, data_dim)
    """

    def __init__(
        self,
        data_dim: int = 256,
        cond_dim: int = 256,
        dim: int = 1024,
        depth: int = 16,
        num_heads: int = 16,
        num_patches: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qk_norm: bool = True,
        use_rope: bool = True,
        time_dim: int = 256,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if data_dim % num_patches != 0:
            raise ValueError(
                f"data_dim({data_dim}) 必须能被 num_patches({num_patches}) 整除"
            )

        self.data_dim = data_dim
        self.num_patches = num_patches
        self.patch_dim = data_dim // num_patches
        self.dim = dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Patchify: 每个 patch 独立投影到 dim
        self.patch_embed = nn.Linear(self.patch_dim, dim)

        # 可学习的 patch 类型 embedding (RoPE 提供相对位置，这里提供 patch 身份)
        self.patch_type_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)

        # 时间编码 → dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # 条件编码 → dim
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qk_norm=qk_norm,
                use_rope=use_rope,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.final_layer = FinalLayer(dim, self.patch_dim)

        self._init_weights()

    def _init_weights(self):
        """基础权重初始化 (AdaLN/FinalLayer 的零初始化在各自模块内完成)"""
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic_init)

        # 重新对 AdaLN / FinalLayer 做零初始化 (apply 会覆盖)
        for block in self.blocks:
            nn.init.zeros_(block.ada_ln[1].weight)
            nn.init.zeros_(block.ada_ln[1].bias)
        nn.init.zeros_(self.final_layer.ada_ln[1].weight)
        nn.init.zeros_(self.final_layer.ada_ln[1].bias)
        nn.init.zeros_(self.final_layer.proj.weight)
        nn.init.zeros_(self.final_layer.proj.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, data_dim) → (B, num_patches, dim)"""
        B = x.size(0)
        patches = x.view(B, self.num_patches, self.patch_dim)
        return self.patch_embed(patches) + self.patch_type_embed

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, num_patches, patch_dim) → (B, data_dim)"""
        return patches.reshape(patches.size(0), self.data_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (batch, data_dim) 加噪状态
            t: (batch,) 时间步 [0, 1]
            condition: (batch, cond_dim) 条件 embedding

        Returns:
            v_pred: (batch, data_dim) 预测速度场
        """
        B = x_t.size(0)

        # 1. Patchify
        tokens = self.patchify(x_t)                      # (B, P, dim)

        # 2. 条件向量: time + condition
        c = self.time_embed(t)                            # (B, dim)
        if condition is not None:
            c = c + self.cond_embed(condition)

        # 3. DiT Blocks
        for block in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, c, use_reentrant=False)
            else:
                tokens = block(tokens, c)

        # 4. 输出层 + Unpatchify
        out = self.final_layer(tokens, c)                 # (B, P, patch_dim)
        return self.unpatchify(out)                       # (B, data_dim)
