"""标准库 bigram 基线：统计当前字符后面出现各字符的次数。

这不是 Transformer，也没有梯度下降；用于先理解训练和生成的区别。
"""

import random
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self):
        self.transitions = defaultdict(Counter)
        self.characters = Counter()

    def fit(self, text):
        if len(text) < 2:
            raise ValueError("训练文本至少需要两个字符")
        self.transitions.clear()
        self.characters = Counter(text)
        for current, following in zip(text, text[1:]):
            self.transitions[current][following] += 1
        return self

    def generate(self, prompt, max_new_tokens=80, seed=42):
        if not self.characters:
            raise ValueError("请先调用 fit 训练模型")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 不能为负数")
        rng = random.Random(seed)
        output = list(prompt)
        for _ in range(max_new_tokens):
            candidates = self.transitions.get(output[-1]) if output else None
            candidates = candidates or self.characters
            next_character = rng.choices(
                list(candidates), weights=list(candidates.values()), k=1
            )[0]
            output.append(next_character)
        return "".join(output)
