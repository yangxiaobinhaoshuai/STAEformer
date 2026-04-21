# Repository Guidelines

## 项目结构与模块组织

本仓库实现 STAEformer，用于交通预测任务。核心模型代码位于 `model/STAEformer.py`，训练入口和实验流程位于 `model/train.py`。通用工具放在 `lib/` 下，包括数据加载（`lib/data_prepare.py`）、评估指标（`lib/metrics.py`）以及日志和随机种子工具（`lib/utils.py`）。数据集以 `data/<DATASET>/data.npz` 和 `data/<DATASET>/index.npz` 的形式存放，当前包含 `METRLA`、`PEMSBAY`、`PEMS03`、`PEMS04`、`PEMS07`、`PEMS08`。训练运行时会生成 `logs/` 和 `saved_models/`。

## 构建、测试与开发命令

使用 uv 安装依赖：

```bash
uv sync
```

训练命令需要在 `model/` 目录执行，因为 `train.py` 依赖 `../data` 和当前目录下的 `STAEformer.yaml`：

```bash
cd model
uv run python train.py -d PEMS08 -g 0
```

`-d` 用于选择数据集，`-g` 用于设置 `CUDA_VISIBLE_DEVICES`。如果 CUDA 不可用，代码会回退到 CPU。可在仓库根目录运行快速语法检查：

```bash
uv run python -m py_compile lib/*.py model/*.py
```

## 编码风格与命名约定

Python 代码使用 4 空格缩进。导入顺序保持为标准库、第三方库、本地模块。沿用现有命名方式：类名使用 `PascalCase`（如 `STAEformer`），函数和变量使用 `snake_case`，配置中的数据集键使用大写。保留有助于理解张量契约的 shape 注释。实验超参数优先写入 `model/STAEformer.yaml`，不要硬编码到 `train.py`。

## 测试指南

当前仓库没有独立测试套件。修改代码后，至少运行上面的语法检查，并用一个熟悉或较小的数据集配置做一次训练冒烟测试。确认 `logs/` 下生成日志、`saved_models/` 下生成 checkpoint，并且最终输出 RMSE/MAE/MAPE。新增测试时放入 `tests/`，文件命名为 `test_<module>.py`。

## 提交与 Pull Request 规范

近期提交信息使用简短的祈使句，部分带有范围前缀，例如 `bugfix:`。提交标题保持简洁，例如 `fix typo` 或 `bugfix: state_dict saving`。Pull Request 应说明行为变化，列出验证过的数据集或命令，并明确是否影响生成文件、日志、checkpoint 或 benchmark 结果。涉及 issue、论文或实验结果时应附上对应引用。

## Agent 专用说明

除非明确要求，不要提交生成的 `logs/` 或 `saved_models/` 内容。保持数据集目录名和“从 `model/` 目录运行训练”的假设不变；如果需要调整，也要同步更新训练命令和文档。
