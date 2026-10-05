"""字符词表：0 保留给未知字符，已知字符按 Unicode 顺序编号。"""

import json
from pathlib import Path


class CharTokenizer:
    unk_id = 0
    unk_token = "<UNK>"

    def __init__(self, vocabulary):
        characters = list(vocabulary)
        if any(not isinstance(char, str) or len(char) != 1 for char in characters):
            raise ValueError("词表中的每一项必须是单个字符")
        self.vocabulary = tuple(sorted(set(characters)))
        self.stoi = {char: index for index, char in enumerate(self.vocabulary, 1)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text):
        return cls(text)

    @property
    def vocab_size(self):
        return len(self.vocabulary) + 1

    def encode(self, text):
        """未知字符逐个映射到 0，不丢字符，也不扩充词表。"""
        return [self.stoi.get(char, self.unk_id) for char in text]

    def decode(self, ids):
        characters = []
        for token_id in ids:
            if type(token_id) is not int or not 0 <= token_id < self.vocab_size:
                raise ValueError(f"无效 token 编号：{token_id!r}")
            characters.append(self.unk_token if token_id == self.unk_id else self.itos[token_id])
        return "".join(characters)

    def save(self, path):
        payload = {"version": 1, "unk_id": self.unk_id, "characters": list(self.vocabulary)}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not isinstance(payload, dict) or type(payload.get("version")) is not int
                or payload["version"] != 1 or type(payload.get("unk_id")) is not int
                or payload["unk_id"] != 0 or not isinstance(payload.get("characters"), list)):
            raise ValueError("无效词表文件：需要 version=1、unk_id=0 和 characters 列表")
        tokenizer = cls(payload["characters"])
        if list(tokenizer.vocabulary) != payload["characters"]:
            raise ValueError("无效词表文件：字符必须唯一且按 Unicode 顺序排列")
        return tokenizer
