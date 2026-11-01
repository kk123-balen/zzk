"""标准库数据接口；从仓库根目录运行 python -m minigpt.data 查看窗口。"""

import argparse
from pathlib import Path

from .tokenizer import CharTokenizer


def make_windows(ids, block_size):
    """返回 [(x, y), ...]，步长为 1；N 个编号产生 N - block_size 个窗口。

    每个窗口需要 block_size + 1 个编号：最后一个编号仅用作 y 的目标。
    """
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
    """返回 (tokenizer, train_windows, val_windows)；仅训练文本参与建词表。

    在编码和切窗口之前分割原文，保证 x 和 y 都不会跨越训练/验证边界。
    """
    train_text, val_text = split_text(text, train_fraction)
    tokenizer = CharTokenizer.from_text(train_text)
    windows = []
    for name, part in (("训练集", train_text), ("验证集", val_text)):
        try:
            windows.append(make_windows(tokenizer.encode(part), block_size))
        except ValueError as exc:
            raise ValueError(f"{name}：{exc}") from exc
    return tokenizer, windows[0], windows[1]


def main():
    parser = argparse.ArgumentParser(description="字符 tokenizer 与训练/验证窗口演示")
    parser.add_argument(
        "--data", type=Path,
        default=Path(__file__).resolve().parents[1] / "data/tiny.txt",
        help="UTF-8 连续文本，默认使用项目自带的 data/tiny.txt",
    )
    parser.add_argument("--block-size", type=int, default=8, help="每个 x/y 的长度，默认 8")
    parser.add_argument("--train-fraction", type=float, default=0.9, help="训练文本比例，默认 0.9")
    parser.add_argument("--vocab-out", type=Path, help="可选：保存词表的 JSON 路径（父目录须已存在）")
    args = parser.parse_args()
    try:
        text = args.data.read_text(encoding="utf-8")
        tokenizer, train, val = prepare_data(text, args.block_size, args.train_fraction)
        if args.vocab_out is not None:
            tokenizer.save(args.vocab_out)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    train_text, val_text = split_text(text, args.train_fraction)
    print(f"训练字符数：{len(train_text)}；验证字符数：{len(val_text)}")
    print(f"词表大小（含未知编号 0）：{tokenizer.vocab_size}")
    print(f"训练窗口数：{len(train)}；验证窗口数：{len(val)}")
    print("未知字符逐个编码为 0，解码显示 <UNK>。")
    for name, windows in (("训练集", train), ("验证集", val)):
        x, y = windows[0]
        print(f"{name}首个窗口：x={x}, y={y}")
        print(f"解码：x={tokenizer.decode(x)!r}, y={tokenizer.decode(y)!r}")
    if args.vocab_out is not None:
        print(f"词表已保存：{args.vocab_out}")


if __name__ == "__main__":
    main()
