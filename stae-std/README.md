# stae-std：可照着重写的 STAEformer

这是原项目的教学版等价实现。它保留了完整主链路：

```text
npz 数据
  → 根据 index 切历史/未来窗口
  → 用训练集统计量标准化输入
  → 输入与三类 embedding 拼接
  → 时间 Transformer
  → 空间 Transformer
  → 一次性预测未来 12 步
  → 原始尺度上训练
  → 验证集早停
  → 最佳模型测试
```

## 文件阅读顺序

1. `config.py`：先知道模型需要哪些参数。
2. `data.py`：弄懂连续序列如何变成 `(B, T, N, C)`。
3. `model.py`：先看 `STAEformer.forward()`，再看 Transformer 和注意力。
4. `metrics.py`：理解为什么要 mask 掉真实值为 0 的位置。
5. `train.py`：把训练、验证、早停、恢复和测试串起来。

`utils.py` 只有随机种子和日志，不是模型重点。

## 运行

在仓库根目录执行：

```bash
uv run python stae-std/train.py -d PEMS08 -g 0
```

只运行一个 epoch，检查整条链路：

```bash
uv run python stae-std/train.py -d PEMS08 -g 0 --max-epochs 1
```

如果没有 CUDA，会自动退回 CPU。PEMS08 的完整模型在 CPU 上也可能较慢。

## 建议手写顺序

第一次不要直接复制全部文件，可以按下面的检查点逐步写：

1. 写 `make_ranges()` 和 `_build_split()`，打印 X、Y shape。
2. 写 `StandardScaler`，确认只有 `x[..., 0]` 发生变化。
3. 写输入投影和三类 embedding，确认拼接结果最后一维是 152。
4. 写单头注意力公式，再扩展为多头。
5. 用 `transpose()` 让同一个 Block 分别沿 time 和 node 工作。
6. 写 mixed projection，确认输出为 `(B, 12, N, 1)`。
7. 先对一个 batch 完成 `forward → loss → backward`。
8. 最后加入完整 epoch、验证、早停、checkpoint 和测试指标。

## 与原项目相比有意做的小整理

- 使用固定默认随机种子，便于复现。
- 删除原代码中未使用的 curriculum learning 和全局变量。
- `predict()` 只 `squeeze(-1)`，避免 batch size 为 1 时误删 batch 维。
- 时间 embedding 的索引增加边界保护。
- 增加输入 shape 检查，让抄写出错时更早暴露问题。

这些整理不改变 STAEformer 的核心建模和训练逻辑。
