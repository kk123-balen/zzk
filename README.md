# MiniGPT：从零开始的学习项目

目标是亲手实现一个小型、仅做文本续写的 decoder-only GPT，并能解释 tokenizer、attention、训练和推理流程。这里的 MiniGPT 是教学项目名称，不是 MiniGPT-4 多模态项目，也不是现成聊天模型。

## 当前已经能做什么

- 使用 Python 标准库运行字符 bigram 模型：统计相邻字符并生成文本。
- 使用字符 tokenizer 编码/解码、保存/加载词表，并构造互不跨界的训练/验证窗口。
- 查看清晰的 GPT 模块边界和逐步实现任务。
- 运行基线测试；提交后 GitHub Actions 自动执行检查。

**任务 1（tokenizer 与数据窗口）已实现；GPT、attention、梯度训练、checkpoint 推理尚未实现。** 占位入口会明确提示未完成。基线输出可能不通顺，这是预期现象。

## 第一次运行（Mac 终端）

需要 Python 3.10 或更高版本。在终端逐行运行：

```bash
git clone https://github.com/kk123-balen/zzk.git
cd zzk
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m minigpt.demo
python -m unittest discover -s tests -v
```

当前不需要安装第三方包、GPU 或租服务器。如果已经下载仓库，进入 zzk 文件夹后从检查 Python 版本开始。使用 VS Code 时打开整个 zzk 文件夹，并选择 .venv 中的 Python 解释器。

换提示词、控制长度：

```bash
python -m minigpt.demo --prompt "学习" --max-new-tokens 40 --seed 42
python -m minigpt.demo --data data/tiny.txt --prompt "Python"
```

默认数据根据模块位置定位；命令从仓库根目录运行。固定 seed 可以重复结果。输入训练集没有的字符也能运行，但会回退到全局字符分布。

## 目录与阅读顺序

| 路径 | 用途 | 状态 |
|---|---|---|
| minigpt/demo.py | 读取文本 → 训练统计模型 → 生成 | 可运行 |
| minigpt/baseline.py | 相邻字符计数和随机采样 | 已实现 |
| minigpt/tokenizer.py | 字符与整数编号转换、词表保存/加载 | 已实现 |
| minigpt/data.py | 先分割文本，再构造训练/验证 x/y 窗口 | 已实现 |
| minigpt/attention.py | 因果 self-attention | 待实现 |
| minigpt/model.py | embedding、Transformer block、输出层 | 待实现 |
| scripts/train.py | batching、loss、反向传播、保存 | 待实现 |
| scripts/generate.py | 加载权重并逐 token 生成 | 待实现 |
| configs/tiny.json | CPU 小模型配置草案 | 待训练入口接入 |
| data/ | 教学语料与数据说明 | 已提供 |
| tests/ | 验证已实现功能 | 已提供 |
| checkpoints/、runs/ | 本地权重与实验日志 | 输出目录 |
| docs/tasks.md | 任务依赖和验收标准 | 学习清单 |

## 学习路线

1. **Python 基础**：列表、字典、循环、函数、类、文件读取。先阅读 demo.py，再阅读 baseline.py；能说明 fit 和 generate 的区别。
2. **Tokenizer 与数据**：手写字符词表，把一句话转成整数列表，再还原。理解输入序列和向右移动一位的目标序列。
3. **PyTorch 基础**：张量形状、矩阵乘法、softmax、autograd、优化器。先在 CPU 跑小例子。
4. **Attention 与 GPT**：单头因果 attention → 多头 → 残差与 LayerNorm → 完整 block。逐层打印形状，不直接复制完整模型。
5. **训练**：先让模型在极小文本上过拟合，再增加独立验证集；记录 train/val loss，保存能恢复的 checkpoint。
6. **推理**：加载 checkpoint，处理上下文长度，比较 temperature 和 top-k 的输出。
7. **AI Infra**：在训练正确后添加设备选择、耗时、tokens/s、内存统计和实验配置记录；有实际瓶颈再考虑 GPU。
8. **Agent**：另外实现工具调用、参数验证和最大执行步数。这个小模型不具备可靠的工具调用能力，Agent 阶段可以使用独立模型接口或测试替身。

每阶段以 [任务与验收标准](docs/tasks.md) 为准，不按固定天数赶进度。完成一个阶段后再进入下一阶段；先把 MiniGPT 的闭环做好，再扩展 Agent 和 Infra。

## 第一个学习动作

运行 demo，打开 baseline.py，找出计数的循环。将 data/tiny.txt 改成多次重复的 `ababab`，把 prompt 改成 `a`，观察生成结果，解释为什么下一个字符可以被预测。

后续任务见仓库 Issues。尚未完成的配置与脚本不能用于 GPT 训练；当前可运行入口始终是 `python -m minigpt.demo`。

## 任务 1：字符 tokenizer 与数据窗口

从仓库根目录运行全部测试（标准库，无需安装依赖）：

```bash
python -m unittest discover -s tests -v
```

接口示例：

```python
from pathlib import Path
from minigpt.tokenizer import CharTokenizer
from minigpt.data import make_windows, prepare_data

tokenizer = CharTokenizer.from_text("abcd")
ids = tokenizer.encode("abcd")  # [1, 2, 3, 4]
assert tokenizer.decode(ids) == "abcd"
windows = make_windows(ids, block_size=2)
# [([1, 2], [2, 3]), ([2, 3], [3, 4])]
tokenizer.save("checkpoints/vocab.json")
loaded = CharTokenizer.load("checkpoints/vocab.json")
assert loaded.encode("abcd") == ids

text = Path("data/tiny.txt").read_text(encoding="utf-8")
tokenizer, train_windows, val_windows = prepare_data(text, block_size=8)
```

- `CharTokenizer(vocabulary)` 接受字符序列；去重后按 Unicode 顺序编号，已知字符从 1 开始，`vocab_size` 包括未知编号 0。
- 未知字符逐个编码为 0，解码显示 `<UNK>`，不删除字符、不自动扩展词表。未知字符的原文无法还原；已知字符可完整往返，包括空格和换行。
- JSON 保存版本、未知编号和有序字符列表；加载会拒绝重复、乱序或不支持的格式，避免编号静默改变。
- `make_windows(ids, block_size)` 返回 `(x, y)` 列表，步长为 1；每个 x/y 长度相同，y 向右偏移一个位置。
- `prepare_data(text, block_size, train_fraction=0.9)` 在 `int(len(text) * train_fraction)` 处先分割原始文本，只用训练文本建词表，再分别编码并构造窗口。验证集的新字符映射为 0；窗口不会跨边界。
- 每一部分至少需要 `block_size + 1` 个字符，否则明确报错。示例语料只有 157 字符，默认 90% 分割后的验证集只有 16 字符，因此这里用 8；配置草案的 `block_size=32` 需要更长语料或调整分割比例。

任务 1 的验收测试见 `tests/test_tokenizer.py` 与 `tests/test_data.py`。接下来按顺序推进任务 2：因果 attention。
