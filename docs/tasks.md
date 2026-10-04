# 实现任务与验收标准

建议每个任务创建独立分支，只实现该阶段能解释清楚的代码。完成后更新 README 状态。每一步都保留标准库演示。

## 1. 字符 tokenizer 与数据窗口

前置：理解列表、字典、类和文件读取。

- 实现 CharTokenizer 的确定性词表、encode(text)、decode(ids)。预留未知字符编号并写明策略，不静默丢字符。
- 支持词表保存与加载，加载后编号必须保持一致。
- 在 minigpt/data.py 实现窗口切分：x 是连续 block_size 个编号，y 是向右偏移一位的编号。
- 先划分训练/验证文本，再分别构造窗口；词表由训练集构造。
- 验收：已知字符 decode(encode(text)) 等于 text；保存加载一致；未知字符行为明确；用 `abcd` 手工检查 x/y；短文本报错清楚；窗口不跨分割边界。

## 2. 因果 attention

前置：任务 1；会 PyTorch 张量、矩阵乘法和 softmax。

- 在自己的虚拟环境安装 PyTorch：`python -m pip install torch`，然后用 `python -c "import torch; print(torch.__version__)"` 验证，并将实际使用的版本记录到依赖文件。
- 先实现函数 causal_self_attention(query, key, value)。输入形状 [batch, time, head_dim]；输出同形状。
- 计算缩放的 QK 转置分数；用上三角掩码阻止看到未来；沿最后一维 softmax 后乘 V。
- 再增加可学习 Q/K/V 投影、多头拆分与合并，要求 n_embd 能被 n_head 整除。
- 验收：用手工小矩阵核对结果；修改未来 token 不影响此前位置输出；概率行和为 1；反向传播有有限梯度；CPU 可运行。

## 3. 组装小型 GPT

前置：任务 2。

- 实现 token embedding 和位置 embedding、pre-norm 残差 attention、MLP、最终 LayerNorm 与 LM head。
- forward 输入整数编号 [B,T]，输出 logits [B,T,V]；先不加入缓存或复杂优化。
- 读取 configs/tiny.json，校验上下文长度、head 数与维度。
- 验收：参数可训练；不同 batch/time 的形状正确；超过上下文长度报错；训练模式关闭 dropout 时满足因果性。

## 4. 训练循环与 checkpoint

前置：任务 1–3。

- scripts/train.py 接收配置、数据路径与输出路径；采用模块方式运行 `python -m scripts.train`。
- 随机取 batch → forward → next-token cross entropy → zero_grad → backward → optimizer.step。
- 设置随机种子，默认 CPU；验证时 eval + no_grad，验证后恢复 train。
- 日志记录 step、train/val loss 与配置。先在小 batch 上证明能过拟合；验证 loss 不保证持续下降。
- 保存模型、optimizer、词表、配置、step 与随机状态；支持恢复，不覆盖无关实验。
- 验收：CPU 短训练无 NaN；重复小 batch 的 loss 明显下降；恢复后能继续训练；独立验证集无窗口泄漏。

## 5. checkpoint 推理脚本

前置：任务 4。

- scripts/generate.py 接收 checkpoint、prompt、max-new-tokens、seed、temperature、top-k；采用 `python -m scripts.generate`。
- 加载同一份词表和配置，使用 eval + no_grad；每次预测最后一个位置并追加 token。
- 只输入最近 block_size 个 token；temperature 必须大于 0，top-k 不超过词表大小。
- 验收：新进程能加载并生成；长 prompt 可处理；同一设备/版本与 seed 下可重复；非法参数有清晰提示；缺失 checkpoint 不伪造输出。

## 后续：Agent 与 AI Infra

完成任务 5 后，增加独立实验：统计 CPU/MPS/CUDA 的训练耗时与 tokens/s；记录依赖、配置与硬件；先测量再优化。Agent 单独设计模型适配器、工具参数校验、可替换的假模型与最大循环次数，避免将教学小模型误当成可用助手。
