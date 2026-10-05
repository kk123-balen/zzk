"""标准库数据接口；先分割文本，再编码和构造 next-token 窗口。"""

from .tokenizer import CharTokenizer


def make_windows(ids, block_size):
    """返回 [(x, y), ...]，步长为 1，x/y 均为整数列表。"""
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size 必须是正整数")
    tokens = list(ids)
    if any(type(token_id) is not int or token_id < 0 for token_id in tokens):
        raise ValueError("token 编号必须是非负整数")
    if len(tokens) < block_size + 1:
        raise ValueError(f"文本太短：至少需要 {block_size + 1} 个字符编号，实际 {len(tokens)} 个")
    return [(tokens[start:start + block_size], tokens[start + 1:start + block_size + 1])
            for start in range(len(tokens) - block_size)]


def split_text(text, train_fraction=0.9):
    """按原始字符位置连续分割，保留空格和换行，不打乱文本。"""
    if (isinstance(train_fraction, bool) or not isinstance(train_fraction, (int, float))
            or not 0 < train_fraction < 1):
        raise ValueError("train_fraction 必须在 0 和 1 之间")
    boundary = int(len(text) * train_fraction)
    return text[:boundary], text[boundary:]


def prepare_data(text, block_size, train_fraction=0.9):
    """返回 (tokenizer, train_windows, val_windows)；仅训练文本参与建词表。"""
    train_text, val_text = split_text(text, train_fraction)
    tokenizer = CharTokenizer.from_text(train_text)
    windows = []
    for name, part in (("训练集", train_text), ("验证集", val_text)):
        try:
            windows.append(make_windows(tokenizer.encode(part), block_size))
        except ValueError as exc:
            raise ValueError(f"{name}：{exc}") from exc
    return tokenizer, windows[0], windows[1]
