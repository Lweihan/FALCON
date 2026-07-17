import argparse
import os
import sys
import glob # 新增导入，用于扫描输出目录
import xml.etree.ElementTree as ET
from multiprocessing import Pool, Process, Queue, cpu_count
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import csv
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# --- 配置与工具函数 ---
csv.field_size_limit(100 * 1024 * 1024)

def extract_section_text(element: ET.Element) -> str:
    parts = []
    def _walk(elem):
        if elem.text and elem.text.strip():
            parts.append(elem.text.strip())
        for child in elem:
            _walk(child)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())
    _walk(element)
    return ' '.join(parts)

def parse_profile_xml(xml_str: str) -> dict:
    result = {'static_information': '', 'interest_profile': '', 'consumption_profile': '', 'demand_targets': ''}
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return result
    
    section_map = {
        'static_information': 'static_information',
        'interest_profile': 'interest_profile',
        'consumption_profile': 'consumption_profile',
        'demand_targets': 'demand_targets'
    }
    for key, tag in section_map.items():
        elem = root.find(tag)
        if elem is not None:
            result[key] = extract_section_text(elem)
    return result

def parse_batch(batch):
    results = []
    for uid, xml_str in batch:
        parsed = parse_profile_xml(xml_str)
        results.append((uid, xml_str, parsed['static_information'], parsed['interest_profile'], parsed['consumption_profile'], parsed['demand_targets']))
    return results

class ProfileEmbedder:
    def __init__(self, model_path: str, gpu_id: int, max_length: int = 512, embed_dim: int = 256):
        self.max_length = max_length
        self.embed_dim = embed_dim
        device = f"cuda:{gpu_id}"
        print(f"[GPU {gpu_id}] 加载模型: {model_path} → {device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2").to(device)
            print(f"[GPU {gpu_id}] ✅ Flash Attention 2 已启用")
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
            print(f"[GPU {gpu_id}] ⚠️ Flash Attention 2 不可用")
        self.model.eval()
        self.device = device

    def encode_field(self, texts: list[str], batch_size: int = 64, field_name: str = "", position: int = 0) -> list[list[float]]:
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        non_empty_texts = [texts[i] for i in non_empty_indices]
        
        desc = f"[GPU {self.device}] {field_name}"
        pbar = tqdm(total=len(non_empty_texts), desc=desc, position=position, leave=True, unit="条", ncols=100)
        
        embeddings = [[] for _ in range(len(texts))]
        
        if non_empty_texts:
            all_embs = []
            for i in range(0, len(non_empty_texts), batch_size):
                batch_texts = non_empty_texts[i:i + batch_size]
                inputs = self.tokenizer(batch_texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    last_token_emb = outputs.last_hidden_state[:, -1]
                    last_token_emb = F.normalize(last_token_emb, p=2, dim=1)
                    batch_embs = last_token_emb[:, :self.embed_dim].cpu().tolist()
                    all_embs.extend(batch_embs)
                pbar.update(len(batch_texts))
        pbar.close()
        
        for i, orig_idx in enumerate(non_empty_indices):
            embeddings[orig_idx] = all_embs[i]
            
        return embeddings

def fast_read_and_parse(input_path: str, parse_workers: int, start_line: int, end_line: int):
    # 1. 先读取表头判断分隔符
    with open(input_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        is_tsv = '\t' in header
    
    raw_rows = []
    with open(input_path, 'r', encoding='utf-8') as f:
        f.readline() # 跳过表头
        # 快速跳过 start_line 之前的行
        for _ in range(start_line):
            if f.readline() == '': break
            
        current_line = start_line
        pbar = tqdm(total=end_line - start_line, desc=f"读取行 {start_line}-{end_line}", unit="行")
        
        # 2. 核心修改：根据 is_tsv 严格区分读取逻辑
        if is_tsv:
            # TSV 模式：直接按行读取并按 \t 分割，避免 csv.reader 的干扰
            for line in f:
                if current_line >= end_line:
                    break
                # 只按第一个 \t 分割，防止 XML 内容中意外包含 \t 导致错位
                parts = line.rstrip('\n').split('\t', 1)
                if len(parts) >= 2:
                    try:
                        raw_rows.append((int(parts[0]), parts[1]))
                    except ValueError:
                        # 容错：如果 ID 不是纯数字，跳过该行并打印警告
                        print(f"\n⚠️ 警告: 跳过非数字ID行 {current_line}: {parts[0][:50]}...")
                current_line += 1
                pbar.update(1)
        else:
            # CSV 模式
            reader = csv.reader(f)
            for line in reader:
                if current_line >= end_line:
                    break
                if len(line) >= 2:
                    try:
                        raw_rows.append((int(line[0]), line[1]))
                    except ValueError:
                        print(f"\n⚠️ 警告: 跳过非数字ID行 {current_line}: {line[0][:50]}...")
                current_line += 1
                pbar.update(1)
                
        pbar.close()

    # 后续的 XML 解析逻辑保持不变
    PARSE_BATCH_SIZE = 2000
    batches = [raw_rows[i:i + PARSE_BATCH_SIZE] for i in range(0, len(raw_rows), PARSE_BATCH_SIZE)]
    del raw_rows
    
    parsed_rows = []
    with Pool(processes=parse_workers) as pool:
        pbar = tqdm(total=len(batches), desc="解析XML", unit="批")
        for batch_results in pool.imap(parse_batch, batches, chunksize=1):
            parsed_rows.extend(batch_results)
            pbar.update(1)
        pbar.close()
        
    return parsed_rows

def process_single_chunk(args):
    chunk_idx, input_path, output_dir, model_path, gpu_id, batch_size, max_length, embed_dim, parse_workers, start_line, end_line = args
    print(f"\n--- 开始处理分片 {chunk_idx} (行 {start_line}-{end_line}) on GPU {gpu_id} ---")
    
    rows_data = fast_read_and_parse(input_path, parse_workers, start_line, end_line)
    if not rows_data:
        return chunk_idx, 0

    embedder = ProfileEmbedder(model_path=model_path, gpu_id=gpu_id, max_length=max_length, embed_dim=embed_dim)
    
    user_ids = [r[0] for r in rows_data]
    standard_texts = [r[1] for r in rows_data]
    static_texts = [r[2] for r in rows_data]
    interest_texts = [r[3] for r in rows_data]
    consumption_texts = [r[4] for r in rows_data]
    demand_texts = [r[5] for r in rows_data]
    del rows_data

    static_features = embedder.encode_field(static_texts, batch_size=batch_size, field_name="static", position=0)
    interest_features = embedder.encode_field(interest_texts, batch_size=batch_size, field_name="interest", position=1)
    consumption_features = embedder.encode_field(consumption_texts, batch_size=batch_size, field_name="consumption", position=2)
    demand_features = embedder.encode_field(demand_texts, batch_size=batch_size, field_name="demand", position=3)

    output_file = os.path.join(output_dir, f"chunk_{chunk_idx:05d}.parquet")
    table = pa.table({
        'user_id': pa.array(user_ids, type=pa.int64()),
        'standard_text': pa.array(standard_texts, type=pa.string()),
        'static_information_feature': pa.array(static_features, type=pa.list_(pa.float64())),
        'interest_profile_feature': pa.array(interest_features, type=pa.list_(pa.float64())),
        'consumption_profile_feature': pa.array(consumption_features, type=pa.list_(pa.float64())),
        'demand_targets_feature': pa.array(demand_features, type=pa.list_(pa.float64())),
    })
    pq.write_table(table, output_file)
    print(f"✅ 分片 {chunk_idx} 处理完成并保存至: {output_file}")

    return chunk_idx, len(user_ids)

# --- Worker 进程函数 ---
def gpu_worker(gpu_id, task_queue, shared_args):
    """ 每个GPU对应一个Worker进程，不断从队列中获取任务并处理。 """
    while True:
        task = task_queue.get()
        if task is None: # 收到停止信号
            break
        # 将任务参数解包，并替换为当前Worker绑定的GPU ID
        chunk_idx, _, output_dir, model_path, _, batch_size, max_length, embed_dim, parse_workers, start_line, end_line = task
        current_task = (chunk_idx, shared_args['input'], shared_args['output_dir'], shared_args['model_path'], gpu_id, shared_args['batch_size'], 512, 256, shared_args['parse_workers'], start_line, end_line)
        process_single_chunk(current_task)

def main():
    parser = argparse.ArgumentParser(description='分片处理用户画像XML')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--chunk_lines', type=int, default=5000, help='每个分片包含的行数')
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--parse_workers', type=int, default=0)
    parser.add_argument('--max_rows', type=int, default=-1, help='最大处理行数，-1表示处理全部')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 核心改动：直接扫描输出目录判断已完成的切片 ---
    completed_chunks = set()
    pattern = os.path.join(args.output_dir, 'chunk_*.parquet')
    existing_files = glob.glob(pattern)
    
    for file_path in existing_files:
        try:
            # 从文件名 chunk_00000.parquet 中提取切片编号
            filename = os.path.basename(file_path)
            chunk_idx = int(filename.split('_')[1].split('.')[0])
            completed_chunks.add(chunk_idx)
        except (IndexError, ValueError):
            continue
            
    if completed_chunks:
        print(f"ℹ️ 检测到 {len(completed_chunks)} 个已完成的切片，将自动跳过: {sorted(list(completed_chunks))}")
    else:
        print("ℹ️ 未检测到已完成的切片，将从头开始处理。")

    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    shared_args = {
        'input': args.input,
        'output_dir': args.output_dir,
        'model_path': args.model_path,
        'batch_size': args.batch_size,
        'parse_workers': args.parse_workers or cpu_count()
    }

    # 启动多卡 Worker 进程
    task_queue = Queue(maxsize=num_gpus * 2)
    workers = []
    for gpu_id in gpu_ids:
        p = Process(target=gpu_worker, args=(gpu_id, task_queue, shared_args))
        p.start()
        workers.append(p)

    print(f"已启动 {num_gpus} 个 GPU Worker，开始生成任务... (最大行数限制: {args.max_rows if args.max_rows != -1 else '无'})")

    # 主进程：按需读取并分发任务
    chunk_idx = 0
    lines_processed = 0
    
    with open(args.input, 'r', encoding='utf-8') as f:
        f.readline() # 跳过表头
        while True:
            if args.max_rows != -1 and lines_processed >= args.max_rows:
                break
            
            # --- 核心改动：跳过已完成的切片 ---
            if chunk_idx in completed_chunks:
                # 快速跳过该分片对应的行数，不放入队列
                for _ in range(args.chunk_lines):
                    if f.readline() == '': break
                lines_processed += args.chunk_lines
                chunk_idx += 1
                continue

            start_line = lines_processed
            current_pos = f.tell()
            # 探测当前分片有多少行
            lines_in_chunk = 0
            for _ in range(args.chunk_lines):
                if f.readline() == '': break
                lines_in_chunk += 1
            if lines_in_chunk == 0:
                break
            f.seek(current_pos)
            end_line = start_line + lines_in_chunk
            lines_processed = end_line
            
            task = (chunk_idx, None, None, None, None, None, None, None, None, start_line, end_line)
            task_queue.put(task)
            chunk_idx += 1

    # 发送停止信号给所有 Worker
    for _ in range(num_gpus):
        task_queue.put(None)
        
    # 等待所有 Worker 完成
    for p in workers:
        p.join()

    print("\n🎉 所有分片处理完成！")
    print(f"请使用以下命令合并所有 Parquet 文件：")
    print(f"python merge_parquet.py --input_dir {args.output_dir} --output_file final_result.parquet")

if __name__ == '__main__':
    main()