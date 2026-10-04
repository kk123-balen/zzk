"""从仓库根目录运行：python -m minigpt.demo。"""

import argparse
from pathlib import Path

from .baseline import BigramModel


def main():
    parser = argparse.ArgumentParser(description="字符级 bigram 入门演示")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data/tiny.txt")
    parser.add_argument("--prompt", default="学习")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens 不能为负数")
    try:
        text = args.data.read_text(encoding="utf-8")
        model = BigramModel().fit(text)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"训练字符数：{len(text)}；不同字符数：{len(model.characters)}")
    print("生成结果（统计基线，不保证语义通顺）：")
    print(model.generate(args.prompt, args.max_new_tokens, args.seed))


if __name__ == "__main__":
    main()
