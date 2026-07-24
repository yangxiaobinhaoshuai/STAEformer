# STAEformer 最简完整学习指南

这份文档只保留理解本项目必须掌握的逻辑。读完后，你应该能回答三个问题：

1. 原始交通序列怎样变成训练样本？
2. STAEformer 怎样从过去 12 步预测未来 12 步？
3. 模型怎样训练、选出最佳参数并完成测试？

---

## 1. 项目只做一件事

给定多个交通传感器过去一段时间的观测，预测所有传感器未来一段时间的交通值。

项目统一使用四维张量：

```text
(B, T, N, C)
```

- `B`：batch size，一批样本数
- `T`：时间步数
- `N`：传感器/节点数
- `C`：特征数

以 PEMS08 为例：

```text
输入 X: (B, 12, 170, 3)
输出 Y: (B, 12, 170, 1)
```

输入的 3 个特征是：

```text
channel 0: 交通观测值
channel 1: 一天中的位置，范围约为 [0, 1)
channel 2: 星期几，取值为 0~6
```

模型的核心思想可以压缩成一句话：

> 把交通值、固定时间信息和可学习的时空信息拼在一起，先沿时间维做 Transformer，再沿节点维做 Transformer，最后预测未来序列。

它不使用道路邻接矩阵或图卷积。节点间关系由空间自注意力学习，数据中难以显式描述的时空模式由 `adaptive_embedding` 学习。

---

## 2. 文件地图

只需要按下面顺序阅读：

```text
model/train.py
    ├── lib/data_prepare.py     数据读取、窗口切片、标准化、DataLoader
    ├── model/STAEformer.py     模型结构和前向传播
    ├── lib/utils.py            scaler、loss、日志、随机种子
    ├── lib/metrics.py          RMSE / MAE / MAPE
    └── model/STAEformer.yaml   各数据集的模型与训练参数
```

数据文件：

```text
data/<DATASET>/data.npz     完整时间序列
data/<DATASET>/index.npz    train/val/test 的窗口边界
```

训练产物：

```text
logs/                       训练日志
saved_models/               验证集最优模型参数
```

---

## 3. 完整执行链路

```text
命令行选择数据集
  → 读取 YAML 配置
  → 构造 STAEformer
  → 读取 data.npz 和 index.npz
  → 切出过去 12 步 X、未来 12 步 Y
  → 只用训练集统计量标准化 X 的交通值
  → 训练：forward → 反标准化预测 → loss → 反向传播
  → 每轮验证并保存内存中的最佳 state_dict
  → 早停后恢复最佳参数
  → 保存 checkpoint
  → 在测试集计算整体及每个预测步的 RMSE/MAE/MAPE
```

下面逐段拆开。

---

## 4. 数据怎样变成样本

入口是 `get_dataloaders_from_index_data()`。

### 4.1 选择特征

`data.npz` 中的 `data` 是完整连续序列。配置默认开启：

```yaml
time_of_day: True
day_of_week: True
```

因此保留通道 `[0, 1, 2]`，即交通值、日内时间、星期。

### 4.2 用索引切窗口

`index.npz` 的每一行是：

```text
[x_start, x_end, y_end]
```

它表示：

```python
x = data[x_start:x_end]  # 历史窗口
y = data[x_end:y_end]    # 未来窗口
```

项目用 `vrange()` 一次生成所有样本的索引。标签只保留交通值：

```python
y = data[y_index][..., :1]
```

所以即使输入有 3 个特征，输出仍只有 1 个目标。

### 4.3 标准化

只用训练集输入的交通值计算：

```text
mean = mean(x_train[..., 0])
std  = std(x_train[..., 0])
```

然后只标准化三个数据集的输入交通值：

```text
x[..., 0] = (x[..., 0] - mean) / std
```

验证集和测试集不能参与均值、标准差的计算，否则会产生数据泄漏。

标签 `Y` 保持原始单位。模型输出先处于标准化尺度，训练和评估前通过：

```text
prediction = prediction * std + mean
```

恢复到原始尺度，再与 `Y` 比较。

### 4.4 DataLoader

- 训练集：`shuffle=True`
- 验证集、测试集：`shuffle=False`
- 三者都返回 `(x_batch, y_batch)`

注意：训练/验证/测试的划分已经写在 `index.npz` 中。YAML 里的 `train_size` 和 `val_size` 在当前代码中没有被读取。

---

## 5. 模型怎样完成一次前向传播

下面始终跟踪：

```text
X: (B, Tin, N, C)
```

默认 `Tin=12`，`Tout=12`。

### 5.1 构造每个“时间—节点”位置的表示

模型先从输入中取出时间标记：

```python
tod = x[..., 1]
dow = x[..., 2]
```

然后构造四类特征。

#### A. 输入投影

```text
原始输入 C=3
  → Linear(3, input_embedding_dim)
  → (B, Tin, N, 24)
```

当前配置的 `input_dim=3`，因此交通值、日内时间和星期三个原始通道都会进入这个线性层。

#### B. 日内时间嵌入

一天被分成 `steps_per_day=288` 个时间槽，即每 5 分钟一个槽：

```text
tod_index = int(tod * 288)
Embedding(288, 24)
  → (B, Tin, N, 24)
```

#### C. 星期嵌入

```text
dow_index = int(dow)
Embedding(7, 24)
  → (B, Tin, N, 24)
```

#### D. 自适应时空嵌入

```text
参数形状: (Tin, N, adaptive_embedding_dim)
默认形状: (12, N, 80)
```

它为每个“历史相对时间位置 × 节点”保存一个可训练向量，再扩展到 batch：

```text
(12, N, 80) → (B, 12, N, 80)
```

这是 STAEformer 最有辨识度的部分：模型直接学习难以由固定时间特征或图结构表达的时空模式。

四类特征在最后一维拼接：

```text
model_dim = 24 + 24 + 24 + 80 = 152
x: (B, Tin, N, 152)
```

代码也支持静态节点嵌入 `spatial_embedding`，但当前所有数据集都设为 `0`，所以没有启用。

### 5.2 时间自注意力

模型执行 `num_layers=3` 层时间 Transformer：

```python
x = temporal_attention(x, dim=1)
```

对每一个节点单独看它的 12 个历史时间步：

```text
固定 B 和 N，在 Tin 维做 self-attention
```

因此它学习“这个节点过去不同时间之间怎样相互影响”。

### 5.3 空间自注意力

接着执行 `num_layers=3` 层空间 Transformer：

```python
x = spatial_attention(x, dim=2)
```

对每一个历史时间步单独看所有节点：

```text
固定 B 和 Tin，在 N 维做 self-attention
```

因此它学习“同一时刻不同传感器之间怎样相互影响”。

顺序是：

```text
全部时间层 → 全部空间层
```

不是时间层和空间层交替执行。

### 5.4 一个 Transformer 层内部

`SelfAttentionLayer` 是标准的 Transformer Encoder 结构：

```text
输入
  → 多头自注意力
  → 残差连接 + LayerNorm
  → 两层前馈网络（Linear → ReLU → Linear）
  → 残差连接 + LayerNorm
```

多头注意力的核心公式：

```text
Attention(Q, K, V) = softmax(QKᵀ / √head_dim)V
```

本项目默认 `mask=False`，所以在给定的 12 个历史步内部可以双向关注；模型仍然看不到未来标签。

`model_dim` 必须能被 `num_heads` 整除。默认 `152 / 4 = 38`，每个头处理 38 维。

### 5.5 输出未来 12 步

默认使用 `use_mixed_proj=True`。模型把每个节点的全部历史时间表示一次性展平：

```text
(B, Tin, N, model_dim)
  → 转置
(B, N, Tin, model_dim)
  → 展平
(B, N, Tin × model_dim)
  → Linear(Tin × model_dim, Tout × output_dim)
  → reshape + transpose
(B, Tout, N, output_dim)
```

它是一次性多步预测，不是把上一步预测再喂回模型的自回归预测。

---

## 6. 模型怎样训练

### 6.1 每个 batch

`train_one_epoch()` 的核心逻辑就是：

```python
prediction_scaled = model(x)
prediction = scaler.inverse_transform(prediction_scaled)
loss = criterion(prediction, y)

optimizer.zero_grad()
loss.backward()
clip_grad_norm_if_enabled()
optimizer.step()
```

关键点：loss 在原始交通值尺度上计算。

### 6.2 损失函数

- `METRLA`、`PEMSBAY`：`MaskedMAELoss`
  - 忽略标签为 0 的位置
  - 避免缺失值影响训练
- `PEMS03/04/07/08`：`HuberLoss`
  - 小误差近似平方损失
  - 大误差近似绝对值损失，对异常值更稳健

### 6.3 优化器和学习率

```text
Adam
  + YAML 中的 lr、weight_decay、eps

MultiStepLR
  + 到达 milestones 指定轮次时
  + lr = lr × lr_decay_rate
```

### 6.4 验证、早停和保存

每轮训练后都在验证集计算 loss：

```text
如果 val_loss 更小：
    记录 best_epoch
    深拷贝 best_state_dict
    wait = 0
否则：
    wait += 1
```

当 `wait >= early_stop` 时停止训练。之后：

1. 恢复验证集最优的 `state_dict`
2. 计算训练集和验证集指标
3. 把最优参数保存到 `saved_models/*.pt`
4. 使用同一个最佳模型测试

保存的是纯 `state_dict`，不是完整模型对象。重新加载时要先用相同配置构造 `STAEformer`，再调用 `load_state_dict()`。

---

## 7. 测试指标

`test_model()` 先得到：

```text
y_true, y_pred: (samples, Tout, N)
```

然后计算：

- RMSE：对较大误差更敏感
- MAE：平均绝对误差，直观稳定
- MAPE：平均绝对百分比误差

代码会输出两类结果：

1. 所有未来时间步合在一起的总体指标
2. 第 1 步到第 12 步各自的指标

三个指标默认都忽略真实值为 0 的位置，并对 mask 重新归一化。

---

## 8. `train.py` 的最简伪代码

整个项目可以浓缩成下面这段：

```python
cfg = load_yaml(dataset)
model = STAEformer(**cfg["model_args"])

train_loader, val_loader, test_loader, scaler = load_data(dataset)
optimizer = Adam(model.parameters(), ...)
scheduler = MultiStepLR(optimizer, ...)

best_state = None
best_val_loss = infinity

for epoch in range(max_epochs):
    model.train()
    for x, y in train_loader:
        pred = scaler.inverse_transform(model(x))
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= early_stop:
            break

model.load_state_dict(best_state)
torch.save(best_state)
test(model, test_loader)
```

而模型本身可以浓缩成：

```python
def forward(x):
    input_feature = input_projection(x)
    tod_feature = tod_embedding(x[..., 1])
    dow_feature = dow_embedding(x[..., 2])
    adaptive_feature = learned_embedding_for_each_time_and_node

    x = concat(
        input_feature,
        tod_feature,
        dow_feature,
        adaptive_feature,
    )

    x = temporal_transformer(x)
    x = spatial_transformer(x)
    return mixed_output_projection(x)
```

---

## 9. 推荐学习顺序

不要一开始逐行读全部代码，按下面五轮学习：

### 第一轮：只跑通数据

阅读 `lib/data_prepare.py`，打印一个 batch，确认：

```text
x.shape == (B, 12, N, 3)
y.shape == (B, 12, N, 1)
```

观察三个输入通道的取值，并手工验证一条 `index.npz` 记录怎样切出 X 和 Y。

### 第二轮：只跟踪 shape

阅读 `STAEformer.forward()`，先不研究注意力公式，只在以下位置打印 shape：

```text
输入
特征拼接后
时间注意力后
空间注意力后
输出
```

### 第三轮：理解注意力

阅读顺序：

```text
SelfAttentionLayer.forward()
  → AttentionLayer.forward()
```

重点理解 `transpose(dim, -2)`：同一套注意力代码通过交换维度，既能沿时间做注意力，也能沿节点做注意力。

### 第四轮：理解训练闭环

按顺序阅读：

```text
train_one_epoch()
  → eval_model()
  → train()
  → predict()
  → test_model()
```

重点确认训练模式、验证模式、反标准化、早停和最佳参数恢复发生在哪里。

### 第五轮：做三个最小实验

1. `adaptive_embedding_dim=0`：观察没有自适应时空嵌入时的变化
2. `num_layers=1`：减少模型深度，观察速度和指标
3. `use_mixed_proj=False`：比较分离式时间投影与混合投影

每次只改一个变量，否则无法判断结果由什么引起。

---

## 10. 运行与检查

安装依赖：

```bash
uv sync
```

训练必须从 `model/` 目录运行：

```bash
cd model
uv run python train.py -d PEMS08 -g 0
```

CUDA 不可用时会自动使用 CPU，但完整训练可能很慢。

语法检查可从仓库根目录运行：

```bash
uv run python -m py_compile lib/*.py model/*.py
```

一次成功的完整训练应当：

- 在 `logs/` 生成日志
- 在 `saved_models/` 生成 `.pt`
- 最终打印 RMSE、MAE、MAPE 和推理时间

---

## 11. 最终心智模型

记住下面这条主线，就已经掌握了整个项目：

```text
连续交通数据
  → index 切成历史窗口和未来窗口
  → 标准化历史交通值
  → 拼接输入、时间和自适应时空嵌入
  → 时间注意力学习单节点的历史依赖
  → 空间注意力学习多节点的相互影响
  → 混合投影一次性生成未来 12 步
  → 反标准化后计算 loss
  → 验证集早停选择最佳模型
  → 测试集报告整体和逐步指标
```

STAEformer 的关键并不是复杂的特殊 Transformer，而是：

> 用简单的时间注意力和空间注意力作为骨架，再用可学习的时空自适应嵌入补充交通数据中隐含的模式。




### Execution cmds

Stae-std
- stae-std/config.py：全部数据集和模型配置
- stae-std/data.py：窗口切片、标准化、DataLoader
- stae-std/model.py：多头注意力、时间/空间 Transformer、STAEformer
- stae-std/metrics.py：Masked MAE、RMSE、MAE、MAPE
- stae-std/train.py：训练、验证、早停、checkpoint 和测试
- stae-std/utils.py：随机种子和日志
- stae-std/README.md：推荐手写顺序和运行说明

关键逻辑处都添加了中文注释，尤其包括：

- [x_start, x_end, y_end] 如何切分样本
- 为什么只能使用训练集计算标准化参数
- 自适应时空嵌入的作用
- 如何用 transpose() 复用时间和空间注意力
- 多头注意力的拆分与还原
- mixed projection 如何一次预测未来12步
- 为什么计算 loss 前需要反标准化
- 验证集早停和最佳参数恢复

已验证：

- Python 语法检查通过
- PEMS08 真实数据切片正确
- 输入 (2, 12, 170, 3) 输出 (2, 12, 170, 1)
- 完整反向传播通过
- 输入投影和自适应嵌入均能获得梯度

train one epoch：
uv run python stae-std/train.py -d PEMS08 -g 0 --max-epochs 1

complete train cmd：
uv run python stae-std/train.py -d PEMS08 -g 0
