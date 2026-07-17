"""
xml_structured.py — JSONL → XML 结构化转换
=============================================
将大模型输出的用户画像 JSON 转为 4-part XML:
  - static_information  (basic_profile + local_scene + other_scene)
  - interest_profile    (兴趣领域评分)
  - consumption_profile (消费特征)
  - demand_targets      (需求目标)
"""

import json
import re
from xml.sax.saxutils import escape as xml_escape
from typing import Optional
import xml.etree.ElementTree as ET


# ============================================================
# 字典/列表 → XML 工具
# ============================================================

def _safe_tag(name: str) -> str:
    """将字段名转为合法 XML tag"""
    name = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '_', str(name))
    if name and (name[0].isdigit() or not (name[0].isalpha() or '\u4e00' <= name[0] <= '\u9fff')):
        name = 'f_' + name
    return name


def _dict_to_xml(tag: str, d: dict, indent: int = 4) -> str:
    spaces = ' ' * indent
    inner = ''
    for k, v in d.items():
        safe_k = _safe_tag(k)
        inner += f'{spaces}<{safe_k}>{xml_escape(str(v))}</{safe_k}>\n'
    if inner:
        return f'{spaces[:-2]}<{tag}>\n{inner}{spaces[:-2]}</{tag}>'
    return f'{spaces[:-2]}<{tag}/>'


def _list_to_xml(tag: str, items: list, item_tag: str = 'item', indent: int = 4) -> str:
    spaces = ' ' * indent
    inner = ''
    for item in items:
        if isinstance(item, dict):
            inner += _dict_to_xml(item_tag, item, indent + 2) + '\n'
        else:
            inner += f'{spaces}<{item_tag}>{xml_escape(str(item))}</{item_tag}>\n'
    if inner:
        return f'{spaces[:-2]}<{tag}>\n{inner}{spaces[:-2]}</{tag}>'
    return f'{spaces[:-2]}<{tag}/>'


# ============================================================
# 单条 record → XML
# ============================================================

def record_to_xml(record: dict, include_time: bool = True) -> str:
    """
    将一条时间步的用户画像 record 转为 XML 字符串。

    XML 结构:
    <user_profile>
      <static_information>
        <basic_profile>...</basic_profile>
        <local_scene>...</local_scene>
        <other_scene>...</other_scene>
      </static_information>
      <interest_profile>...</interest_profile>
      <consumption_profile>...</consumption_profile>
      <demand_targets>...</demand_targets>
    </user_profile>
    """
    lines = ['<user_profile>']

    if include_time and 'time' in record:
        lines.append(f'  <time>{xml_escape(str(record["time"]))}</time>')

    # --- static_information ---
    lines.append('  <static_information>')
    bp = record.get('basic_profile', {})
    lines.append(_dict_to_xml('basic_profile', bp, indent=4) if bp else '    <basic_profile/>')

    ls = record.get('local_scene', {})
    lines.append(_dict_to_xml('local_scene', ls, indent=4) if ls else '    <local_scene/>')

    os_ = record.get('other_scene', {})
    lines.append(_dict_to_xml('other_scene', os_, indent=4) if os_ else '    <other_scene/>')
    lines.append('  </static_information>')

    # --- interest_profile ---
    ip = record.get('interest_profile', [])
    lines.append(_list_to_xml('interest_profile', ip, 'item', indent=2) if ip else '  <interest_profile/>')

    # --- consumption_profile ---
    cp = record.get('consumption_profile', {})
    lines.append('  <consumption_profile>')
    cap = cp.get('能力层级', cp.get('capacity_level', ''))
    lines.append(f'    <能力层级>{xml_escape(str(cap))}</能力层级>')
    pred = cp.get('消费预测', cp.get('consumption_prediction', []))
    lines.append(_list_to_xml('消费预测', pred, 'item', indent=4) if pred else '    <消费预测/>')
    local = cp.get('本地活跃', cp.get('local_active', []))
    lines.append(_list_to_xml('本地活跃', local, 'item', indent=4) if local else '    <本地活跃/>')
    lines.append('  </consumption_profile>')

    # --- demand_targets ---
    dt = record.get('demand_targets', [])
    lines.append(_list_to_xml('demand_targets', dt, 'item', indent=2) if dt else '  <demand_targets/>')

    lines.append('</user_profile>')
    return '\n'.join(lines)


# ============================================================
# XML Section 分段提取（用于分部分 embedding）
# ============================================================

def extract_section_text(element: ET.Element) -> str:
    """递归提取 XML 元素下所有文本内容"""
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
    """将 XML 字符串解析回 4 个 section 的文本"""
    result = {
        'static_information': '',
        'interest_profile': '',
        'consumption_profile': '',
        'demand_targets': ''
    }
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


# ============================================================
# JSONL 文件加载 & 用户转换
# ============================================================

def load_jsonl(file_path: str, max_lines: Optional[int] = None) -> list[dict]:
    """加载 JSONL 文件，每行一个用户"""
    users = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            users.append(json.loads(line))
    return users


def user_to_xml_sections(user_data: dict) -> list[dict]:
    """
    将一个用户数据转为按时间排序的 XML sections 列表。

    返回: list of {
        "time": str,
        "xml_full": str,
        "static_information": str,
        "interest_profile": str,
        "consumption_profile": str,
        "demand_targets": str,
    }
    """
    records = sorted(user_data.get('records', []), key=lambda r: r.get('time', ''))

    sections = []
    for rec in records:
        xml_str = record_to_xml(rec)
        parsed = parse_profile_xml(xml_str)
        sections.append({
            'time': rec.get('time', ''),
            'xml_full': xml_str,
            'static_information': parsed['static_information'],
            'interest_profile': parsed['interest_profile'],
            'consumption_profile': parsed['consumption_profile'],
            'demand_targets': parsed['demand_targets'],
        })
    return sections
