#!/usr/bin/env python3
"""
用户画像槽位提取与XML转换（多进程版 + tqdm进度条）

从parquet文件中读取profile_text字段，提取结构化信息并转换为XML格式，
与user_id组合后输出为CSV。

优化点:
  1. multiprocessing.Pool 多进程并行处理（CPU密集型，无需GPU）
  2. tqdm 进度条实时显示处理进度
  3. 生成器逐行读取，避免 to_pylist() 一次性加载全部数据到内存

用法:
  python standard_mp.py --input data.parquet --rows 100 --output result.csv
  python standard_mp.py --input data.parquet --rows all --output result.csv --workers 16
"""

import csv
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from xml.sax.saxutils import escape as xml_escape

import pyarrow.parquet as pq
from tqdm import tqdm

# CSV 大字段支持
csv.field_size_limit(100 * 1024 * 1024)


# ============================================================
# 文本预处理
# ============================================================

def clean_result(text: str) -> str:
    """清理profile_text字段中的无用前缀和噪声"""
    text = text.strip()
    text = re.sub(r'^---\s*', '', text)
    text = re.sub(r'(\d+)/1\s+(\d)', r'\1/1\2', text)
    text = re.sub(r' {3,}', ' ', text)
    return text.strip()


def split_sections(text: str) -> tuple:
    """将文本分割为Section1和Section2"""
    match = re.split(r'\s*---\s*##\s*Section2', text)
    if len(match) >= 2:
        section1 = match[0].strip()
        section2 = '## Section2' + match[1].strip()
        return section1, section2

    parts = re.split(r'\s*---\s*', text)
    section1, section2 = '', ''
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '## Section1' in part or '[用户生活及场景]' in part:
            section1 = part
        elif '## Section2' in part or '需求目标' in part:
            if not part.startswith('##'):
                part = '## Section2：' + part
            section2 = part

    return section1, section2


# ============================================================
# Section1 解析：用户画像
# ============================================================

def parse_basic_profile(section1: str) -> dict:
    """提取基础画像"""
    profile = {}
    match = re.search(r'1\.基础画像[：:]\s*(.*?)(?=\s*2\.本地场景)', section1, re.DOTALL)
    if not match:
        return profile

    content = match.group(1).strip()
    fields = re.split(r'\s*\|\s*', content)

    for field in fields:
        field = field.strip()
        if not field:
            continue
        m = re.match(r'([^：:]+)[：:]\s*(.*)', field)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            profile[key] = value

    return profile


def _parse_dash_items(text: str) -> dict:
    """解析以 '- key：value' 格式排列的子项"""
    result = {}
    items = re.findall(r'-\s*([^：:]+?)[：:]\s*(.*?)(?=\s+-\s|$)', text, re.DOTALL)

    for key, value in items:
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value

    return result


def parse_local_scene(section1: str) -> dict:
    """提取本地场景"""
    match = re.search(
        r'2\.本地场景[（(].*?[）)][：:]\s*(.*?)(?=\s*3\.其他场景特征)',
        section1, re.DOTALL
    )
    if not match:
        return {}

    return _parse_dash_items(match.group(1))


def parse_other_scene(section1: str) -> dict:
    """提取其他场景特征"""
    match = re.search(
        r'3\.其他场景特征[：:]\s*(.*?)(?=\[内容兴趣图谱\]|\[消费特征\]|## Section)',
        section1, re.DOTALL
    )
    if not match:
        return {}

    return _parse_dash_items(match.group(1))


def parse_interest_profile(section1: str) -> list:
    """提取兴趣领域评分列表"""
    result = []

    match = re.search(
        r'\[内容兴趣图谱\]\s*1\.兴趣领域评分.*?[：:]\s*(.*?)(?=\[消费特征\]|## Section|$)',
        section1, re.DOTALL
    )
    if not match:
        return result

    content = match.group(1).strip()
    items = re.findall(
        r'([\u4e00-\u9fffa-zA-Z0-9/+()（）和\-]+)[：:]\s*(\d+/10)',
        content
    )

    for domain, score in items:
        result.append({
            '领域': domain.strip(),
            '评分': score.strip()
        })

    return result


def parse_consumption_profile(section1: str) -> dict:
    """提取消费特征"""
    result = {'能力层级': '', '消费预测': [], '本地活跃': []}

    match = re.search(
        r'\[消费特征\]\s*(.*?)(?=\s*---|## Section|$)',
        section1, re.DOTALL
    )
    if not match:
        return result

    content = match.group(1).strip()

    cap_match = re.search(r'能力层级[：:]\s*(\S+)', content)
    if cap_match:
        result['能力层级'] = cap_match.group(1).strip().rstrip('|').strip()

    pred_match = re.search(
        r'2\.可能履约的消费领域及出价预测.*?[：:]\s*(.*?)(?=3\.本地消费高活跃领域|$)',
        content, re.DOTALL
    )
    if pred_match:
        pred_content = pred_match.group(1).strip()

        group_starts = list(re.finditer(
            r'[\u4e00-\u9fff][\u4e00-\u9fff\w/+()（）\-]*[：:]\s*\d+/10',
            pred_content
        ))

        for i, start_match in enumerate(group_starts):
            end_pos = group_starts[i + 1].start() if i + 1 < len(group_starts) else len(pred_content)
            group_text = pred_content[start_match.start():end_pos].strip()

            parts = re.split(r'\s*\|\s*', group_text, maxsplit=2)

            item = {}

            if parts:
                m = re.match(r'(.+?)[：:]\s*(\d+/10)', parts[0].strip())
                if m:
                    item['领域'] = m.group(1).strip()
                    item['评分'] = m.group(2).strip()

            if len(parts) >= 2:
                second = parts[1].strip()
                m = re.match(r'出价区间[：:]\s*(.*)', second)
                if m:
                    item['出价区间'] = m.group(1).strip()
                else:
                    item['出价区间'] = second

            if len(parts) >= 3:
                third = parts[2].strip()
                if third:
                    m = re.match(r'消费习惯[：:]\s*(.*)', third)
                    if m:
                        item['消费习惯'] = m.group(1).strip()
                    else:
                        item['消费习惯'] = third

            if item.get('领域'):
                result['消费预测'].append(item)

    local_match = re.search(
        r'3\.本地消费高活跃领域.*?[：:]\s*(.*?)(?=\s*---|## Section|$)',
        content, re.DOTALL
    )
    if local_match:
        local_content = local_match.group(1).strip()
        if '暂无' in local_content or '无明显' in local_content:
            return result
        local_items = re.findall(
            r'([\u4e00-\u9fffa-zA-Z0-9/+()（）和\-]+)[：:]\s*(\d+/10)',
            local_content
        )
        for domain, score in local_items:
            result['本地活跃'].append({
                '领域': domain.strip(),
                '评分': score.strip()
            })

    return result


# ============================================================
# Section2 解析：需求目标
# ============================================================

def parse_demand_targets(section2: str) -> list:
    """提取需求目标列表"""
    result = []

    if not section2:
        return result

    cleaned = re.sub(
        r'本Section基于.*?类型权重\s*',
        '', section2, flags=re.DOTALL
    )
    cleaned = re.sub(
        r'深转潜力TOP\d+行业.*?不列入）[：:]\s*',
        '', cleaned, flags=re.DOTALL
    )

    targets = re.findall(
        r'【行业\d+】(.+?)\s+深转潜力[：:]\s*(\d+/10)\s+兴趣细分锚点[：:]\s*(.*?)(?=【行业\d+】|$)',
        cleaned, re.DOTALL
    )

    for industry, score, anchor in targets:
        industry = industry.strip()
        score = score.strip()
        anchor = anchor.strip()

        item = {
            '行业': industry,
            '深转潜力': score,
            '兴趣细分锚点': anchor
        }

        inc_match = re.search(r'\s+增量信息[：:]\s*(.*)', anchor)
        if inc_match:
            item['兴趣细分锚点'] = anchor[:inc_match.start()].strip()
            item['增量信息'] = inc_match.group(1).strip()

        result.append(item)

    return result


# ============================================================
# 槽位提取 & XML 转换
# ============================================================

def extract_slots(result_text: str) -> dict:
    """从profile_text文本中提取结构化槽位"""
    cleaned = clean_result(result_text)
    section1, section2 = split_sections(cleaned)

    slots = {
        'basic_profile': parse_basic_profile(section1) if section1 else {},
        'local_scene': parse_local_scene(section1) if section1 else {},
        'other_scene': parse_other_scene(section1) if section1 else {},
        'interest_profile': parse_interest_profile(section1) if section1 else [],
        'consumption_profile': (
            parse_consumption_profile(section1) if section1
            else {'能力层级': '', '消费预测': [], '本地活跃': []}
        ),
        'demand_targets': parse_demand_targets(section2)
    }

    return slots


def _dict_to_xml(tag: str, d: dict, indent: int = 2) -> str:
    """将字典转换为XML"""
    spaces = ' ' * indent
    inner = ''
    for k, v in d.items():
        safe_k = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '_', k)
        if safe_k[0].isdigit() or not safe_k[0:1].isalpha() and not safe_k[0:1].isupper() and not '\u4e00' <= safe_k[0] <= '\u9fff':
            safe_k = 'f_' + safe_k
        inner += f'{spaces}<{safe_k}>{xml_escape(str(v))}</{safe_k}>\n'
    if inner:
        return f'{spaces[:-2]}<{tag}>\n{inner}{spaces[:-2]}</{tag}>'
    else:
        return f'{spaces[:-2]}<{tag}/>'


def _list_to_xml(tag: str, items: list, item_tag: str, indent: int = 2) -> str:
    """将列表转换为XML"""
    spaces = ' ' * indent
    inner = ''
    for item in items:
        if isinstance(item, dict):
            inner += _dict_to_xml(item_tag, item, indent + 2) + '\n'
        else:
            inner += f'{spaces}<{item_tag}>{xml_escape(str(item))}</{item_tag}>\n'
    if inner:
        return f'{spaces[:-2]}<{tag}>\n{inner}{spaces[:-2]}</{tag}>'
    else:
        return f'{spaces[:-2]}<{tag}/>'


def slots_to_xml(slots: dict) -> str:
    """将提取的槽位字典转换为XML字符串"""
    lines = ['<user_profile>']

    lines.append('  <static_information>')
    bp = slots.get('basic_profile', {})
    if bp:
        lines.append(_dict_to_xml('basic_profile', bp, indent=4))
    else:
        lines.append('    <basic_profile/>')

    ls = slots.get('local_scene', {})
    if ls:
        lines.append(_dict_to_xml('local_scene', ls, indent=4))
    else:
        lines.append('    <local_scene/>')

    os_ = slots.get('other_scene', {})
    if os_:
        lines.append(_dict_to_xml('other_scene', os_, indent=4))
    else:
        lines.append('    <other_scene/>')

    lines.append('  </static_information>')

    ip = slots.get('interest_profile', [])
    if ip:
        lines.append(_list_to_xml('interest_profile', ip, 'item', indent=2))
    else:
        lines.append('  <interest_profile/>')

    cp = slots.get('consumption_profile', {'能力层级': '', '消费预测': [], '本地活跃': []})
    lines.append('  <consumption_profile>')
    cap = cp.get('能力层级', '')
    lines.append(f'    <能力层级>{xml_escape(cap)}</能力层级>')
    pred = cp.get('消费预测', [])
    if pred:
        lines.append(_list_to_xml('消费预测', pred, 'item', indent=4))
    else:
        lines.append('    <消费预测/>')
    local = cp.get('本地活跃', [])
    if local:
        lines.append(_list_to_xml('本地活跃', local, 'item', indent=4))
    else:
        lines.append('    <本地活跃/>')
    lines.append('  </consumption_profile>')

    dt = slots.get('demand_targets', [])
    if dt:
        lines.append(_list_to_xml('demand_targets', dt, 'item', indent=2))
    else:
        lines.append('  <demand_targets/>')

    lines.append('</user_profile>')
    return '\n'.join(lines)


# ============================================================
# 多进程 Worker
# ============================================================

def process_row(args: tuple) -> tuple:
    """
    处理单行数据：提取槽位 + 转XML。

    Args:
        args: (user_id, profile_text)

    Returns:
        (user_id, profile_xml, error_msg)
        - 成功: (user_id, xml_str, None)
        - 失败: (user_id, error_xml, error_msg)
    """
    uid, text = args
    text = str(text) if text else ''

    try:
        slots = extract_slots(text)
        xml_str = slots_to_xml(slots)
        return (uid, xml_str, None)
    except Exception as e:
        error_xml = f'<user_profile><error>{xml_escape(str(e))}</error></user_profile>'
        return (uid, error_xml, str(e))


# ============================================================
# 主流程
# ============================================================

def main(input_parquet: str, rows: str, output_csv: str, workers: int):
    """
    主函数：读取parquet → 多进程提取槽位 → 写CSV。

    Args:
        input_parquet: 输入parquet文件路径
        rows: 处理行数，数字字符串或 'all'
        output_csv: 输出CSV文件路径
        workers: 并行进程数
    """
    if not os.path.exists(input_parquet):
        print(f"错误: 输入文件不存在: {input_parquet}", file=sys.stderr)
        sys.exit(1)

    # 解析行数
    if rows.lower() == 'all':
        max_rows = None
    else:
        try:
            max_rows = int(rows)
        except ValueError:
            print(f"错误: 无效的行数参数: {rows}（应为数字或 'all'）", file=sys.stderr)
            sys.exit(1)

    print(f"输入文件: {input_parquet}")
    print(f"处理行数: {'全部' if max_rows is None else max_rows}")
    print(f"并行进程: {workers}")
    print(f"输出文件: {output_csv}")

    # 读取parquet
    print("\n读取parquet文件...")
    table = pq.read_table(input_parquet)
    total_in_file = table.num_rows
    print(f"文件总行数: {total_in_file}")

    if max_rows is not None:
        table = table.slice(0, min(max_rows, total_in_file))
        process_rows = min(max_rows, total_in_file)
    else:
        process_rows = total_in_file

    print(f"本次处理行数: {process_rows}")

    # 创建输出目录
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 用生成器逐行产出，避免 to_pylist() 一次性加载全部数据到内存
    def row_generator():
        for i in range(process_rows):
            uid = table.column("user_id")[i].as_py()
            text = table.column("profile_text")[i].as_py()
            yield (uid, text)

    # 多进程处理 + tqdm
    print(f"\n🚀 启动 {workers} 个进程并行处理...\n")

    total = 0
    error_count = 0

    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'profile_xml'])

        with Pool(processes=workers) as pool:
            pbar = tqdm(
                pool.imap(process_row, row_generator(), chunksize=max(1, process_rows // (workers * 4))),
                total=process_rows,
                desc="提取槽位&转XML",
                unit="行",
                ncols=100,
            )

            for uid, xml_str, error_msg in pbar:
                writer.writerow([uid, xml_str])
                total += 1

                if error_msg:
                    error_count += 1
                    pbar.set_postfix({"错误": error_count})

    print(f"\n✅ 处理完成: 共 {total} 行, {error_count} 个错误")
    print(f"CSV已保存到: {output_csv}")

    # 输出示例供验证
    if total > 0:
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            first = next(reader)
            print(f"\n--- 示例 (user_id={first['user_id']}) ---")
            print(first['profile_xml'][:2000])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='用户画像槽位提取与XML转换（多进程版）'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='输入parquet文件路径（含user_id和profile_text两列）')
    parser.add_argument('--rows', type=str, default='all',
                        help='处理行数: 数字或 "all" 处理全部（默认: all）')
    parser.add_argument('--output', type=str, required=True,
                        help='输出CSV文件路径（含user_id和profile_xml两列）')
    parser.add_argument('--workers', type=int, default=0,
                        help='并行进程数（默认: CPU核心数）')

    args = parser.parse_args()

    # 自动检测 CPU 核心数，确保为整数
    if args.workers <= 0:
        args.workers = cpu_count() / 2
    args.workers = min(52, int(args.workers))

    main(input_parquet=args.input, rows=args.rows, output_csv=args.output, workers=args.workers)