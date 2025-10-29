#!/usr/bin/env python3
"""
ABC Notation 结构位掩码工具

任务: NG-1
目标: 区分 ABC notation 中的头部行、主体结构位、其它字符
日期: 2025-10-21

关键设计:
1. 头部行 (M:, L:, Q:, K:, V:, etc.) → 不计分/不反转
2. 主体结构位 (|, |:, :|, 换行等) → 参与 Top-k 计分
3. 其它主体字符 (音符、时长) → 可选参与窗口统计
"""

import re
from typing import Tuple, List, Dict
import numpy as np

# ABC notation 头部字段标识符 (完整列表)
HEADER_PREFIXES = [
    'X:', 'T:', 'C:', 'Z:',  # 基本元信息
    'M:', 'L:', 'Q:', 'K:',  # 节拍、单位长度、速度、调性
    'V:', 'P:', 'I:', 'U:',  # 声部、部分、指令、用户自定义
    'A:', 'B:', 'D:', 'F:',  # 区域、书籍、修饰、文件名
    'G:', 'H:', 'N:', 'O:',  # 组、历史、注释、起源
    'R:', 'S:', 'W:', 'w:',  # 节奏、源、歌词
    '%%',                     # 格式化指令
]

# 主体结构位字符 (需要参与 Top-k 计分的)
STRUCTURE_CHARS = [
    '|',   # 普通小节线
    ':',   # 重复标记的一部分
    '[',   # 和弦/重复标记开始
    ']',   # 和弦/重复标记结束
    '\n',  # 换行符 (结构边界)
]

# 结构位模式 (用于更精确的匹配)
STRUCTURE_PATTERNS = [
    r'\|:',    # 重复开始
    r':\|',    # 重复结束
    r'\|\|',   # 双小节线
    r'\|\]',   # 结尾小节线
    r'\[\|',   # 开始小节线
    r'\|',     # 单小节线 (放在最后，避免被前面的匹配)
]


def is_header_line(line: str) -> bool:
    """
    判断一行是否为 ABC 头部行
    
    Args:
        line: ABC notation 的一行文本
    
    Returns:
        True 如果是头部行，False 否则
    """
    line_stripped = line.strip()
    if not line_stripped:
        return False
    
    # 检查是否以任何头部标识符开头
    for prefix in HEADER_PREFIXES:
        if line_stripped.startswith(prefix):
            return True
    
    # 特殊情况: 空行或只有空格
    if not line_stripped:
        return True
    
    return False


def find_structure_positions(text: str, include_header_boundary: bool = True) -> np.ndarray:
    """
    查找文本中所有结构位的位置
    
    Args:
        text: ABC notation 完整文本
        include_header_boundary: 是否在头部/主体分界处标记一个结构位
    
    Returns:
        结构位的字符索引数组 (1D numpy array)
    """
    struct_positions = []
    
    # 1. 找到头部/主体分界点
    lines = text.split('\n')
    header_end_pos = 0
    in_header = True
    
    for i, line in enumerate(lines):
        if is_header_line(line):
            header_end_pos += len(line) + 1  # +1 for '\n'
        else:
            in_header = False
            break
    
    if include_header_boundary and header_end_pos > 0:
        struct_positions.append(header_end_pos - 1)  # 标记头部结束位置
    
    # 2. 在主体部分查找结构位
    body_text = text[header_end_pos:]
    body_start = header_end_pos
    
    # 2.1 查找换行符 (结构边界)
    for match in re.finditer(r'\n', body_text):
        struct_positions.append(body_start + match.start())
    
    # 2.2 查找小节线及相关结构
    for pattern in STRUCTURE_PATTERNS:
        for match in re.finditer(pattern, body_text):
            # 记录匹配范围内的所有字符位置
            for pos in range(match.start(), match.end()):
                abs_pos = body_start + pos
                if abs_pos not in struct_positions:
                    struct_positions.append(abs_pos)
    
    return np.array(sorted(struct_positions), dtype=np.int64)


def abc_structure_mask(
    abc_text: str,
    char_nlls: np.ndarray = None,
    return_spans: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    为 ABC notation 文本生成结构位掩码
    
    Args:
        abc_text: ABC notation 完整文本
        char_nlls: 可选，每字符的 NLL 数组 (用于验证长度)
        return_spans: 是否返回头部/主体的范围信息
    
    Returns:
        structure_mask: 布尔数组，True 表示结构位，False 表示非结构位
        info_dict: 包含以下信息的字典
            - 'header_end': 头部结束位置
            - 'structure_positions': 结构位的索引数组
            - 'n_header_chars': 头部字符数
            - 'n_body_chars': 主体字符数
            - 'n_structure_chars': 结构位字符数
            - 'structure_ratio': 结构位占主体的比例
            (如果 return_spans=True)
            - 'header_spans': [(start, end), ...] 头部行的范围
            - 'body_spans': [(start, end), ...] 主体行的范围
    """
    text_len = len(abc_text)
    
    # 初始化掩码 (全部为 False)
    structure_mask = np.zeros(text_len, dtype=bool)
    
    # 找到头部/主体分界点
    lines = abc_text.split('\n')
    header_end_pos = 0
    header_lines = []
    body_lines = []
    
    current_pos = 0
    in_header = True
    
    for i, line in enumerate(lines):
        line_start = current_pos
        line_end = current_pos + len(line)
        
        if in_header and is_header_line(line):
            header_lines.append((line_start, line_end))
            header_end_pos = line_end + 1  # +1 for '\n'
        else:
            in_header = False
            body_lines.append((line_start, line_end))
        
        current_pos = line_end + 1  # +1 for '\n'
    
    # 查找结构位
    struct_positions = find_structure_positions(abc_text, include_header_boundary=False)
    
    # 设置掩码
    structure_mask[struct_positions] = True
    
    # 统计信息
    n_header_chars = header_end_pos
    n_body_chars = text_len - header_end_pos
    n_structure_chars = len(struct_positions)
    structure_ratio = n_structure_chars / max(n_body_chars, 1)
    
    info_dict = {
        'header_end': header_end_pos,
        'structure_positions': struct_positions,
        'n_header_chars': n_header_chars,
        'n_body_chars': n_body_chars,
        'n_structure_chars': n_structure_chars,
        'structure_ratio': structure_ratio,
    }
    
    if return_spans:
        info_dict['header_spans'] = header_lines
        info_dict['body_spans'] = body_lines
    
    # 验证长度一致性 (如果提供了 char_nlls)
    if char_nlls is not None:
        assert len(char_nlls) == text_len, \
            f"NLL 长度 ({len(char_nlls)}) 与文本长度 ({text_len}) 不匹配"
    
    return structure_mask, info_dict


def extract_structure_nlls(
    char_nlls: np.ndarray,
    structure_mask: np.ndarray,
    header_end: int
) -> np.ndarray:
    """
    提取主体部分的结构位 NLL
    
    Args:
        char_nlls: 每字符的 NLL 数组
        structure_mask: 结构位掩码
        header_end: 头部结束位置
    
    Returns:
        主体结构位的 NLL 数组
    """
    # 只取主体部分
    body_nlls = char_nlls[header_end:]
    body_mask = structure_mask[header_end:]
    
    # 提取结构位 NLL
    structure_nlls = body_nlls[body_mask]
    
    return structure_nlls


def visualize_structure_mask(
    abc_text: str,
    structure_mask: np.ndarray,
    max_lines: int = 20
) -> str:
    """
    可视化结构位掩码 (用于调试和验证)
    
    Args:
        abc_text: ABC notation 文本
        structure_mask: 结构位掩码
        max_lines: 最多显示多少行
    
    Returns:
        可视化字符串
    """
    lines = abc_text.split('\n')[:max_lines]
    viz_lines = []
    
    current_pos = 0
    for line in lines:
        # 原始行
        viz_lines.append(line)
        
        # 掩码行 (用 ^ 标记结构位)
        mask_line = ''
        for i, char in enumerate(line):
            pos = current_pos + i
            if pos < len(structure_mask) and structure_mask[pos]:
                mask_line += '^'
            else:
                mask_line += ' '
        viz_lines.append(mask_line)
        
        current_pos += len(line) + 1  # +1 for '\n'
    
    return '\n'.join(viz_lines)


# ============================================================================
# 单元测试用例
# ============================================================================

def test_abc_structure_mask():
    """
    单元测试: ABC 结构位掩码
    """
    test_abc = """X:1
T:Test Piece
M:4/4
L:1/8
Q:1/4=120
K:C
[V:1] treble
|: C2 D2 E2 F2 | G8 | F2 E2 D2 C2 | C8 :|
|: G2 A2 B2 c2 | d8 | c2 B2 A2 G2 | G8 :|
"""
    
    print("="*80)
    print("ABC 结构位掩码单元测试")
    print("="*80)
    
    # 生成掩码
    structure_mask, info = abc_structure_mask(test_abc, return_spans=True)
    
    print(f"\n文本长度: {len(test_abc)}")
    print(f"头部结束位置: {info['header_end']}")
    print(f"头部字符数: {info['n_header_chars']}")
    print(f"主体字符数: {info['n_body_chars']}")
    print(f"结构位字符数: {info['n_structure_chars']}")
    print(f"结构位比例: {info['structure_ratio']:.2%}")
    
    print(f"\n头部行数: {len(info['header_spans'])}")
    print(f"主体行数: {len(info['body_spans'])}")
    
    print(f"\n前 20 个结构位: {info['structure_positions'][:20]}")
    
    # 可视化
    print("\n" + "="*80)
    print("结构位可视化 (^ 标记结构位):")
    print("="*80)
    viz = visualize_structure_mask(test_abc, structure_mask)
    print(viz)
    
    # 验证: 头部行不应有结构位标记
    header_text = test_abc[:info['header_end']]
    header_mask = structure_mask[:info['header_end']]
    assert not np.any(header_mask), "❌ 头部行包含结构位标记 (应全部为 False)"
    print("\n✅ 验证通过: 头部行不包含结构位标记")
    
    # 验证: 主体部分应有结构位
    body_mask = structure_mask[info['header_end']:]
    assert np.any(body_mask), "❌ 主体部分没有结构位标记"
    print("✅ 验证通过: 主体部分包含结构位标记")
    
    # 验证: 小节线位置
    body_text = test_abc[info['header_end']:]
    barline_count = body_text.count('|')
    print(f"\n小节线数量 (目测): {barline_count}")
    print(f"结构位数量 (实际): {info['n_structure_chars']}")
    
    print("\n" + "="*80)
    print("✅ 单元测试完成!")
    print("="*80)
    
    return structure_mask, info


if __name__ == "__main__":
    test_abc_structure_mask()

