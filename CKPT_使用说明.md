# 断点续跑与临时存储使用说明

## 概述

`eval/collect_meta_routes.py` 现在支持可选的**断点续跑（checkpointing）**功能。这个功能允许你：

1. **临时存储**：在收集过程中定期将已收集的轨迹保存到磁盘，避免因程序崩溃或手动退出导致数据丢失
2. **断点续跑**：如果程序中断，可以从上次保存的进度继续收集，不会重复已收集的种子
3. **部分数据导出**：即使还没收集完目标数量，也可以将已收集的数据导出为标准 `routes.npz` 格式，直接用于 `analysis/pkd_cycle_sampler.py` 等分析脚本

## 快速开始

### 1. 启用 checkpointing

在运行 `collect_meta_routes.py` 时，添加以下参数：

```bash
python eval/collect_meta_routes.py \
    --ckpt_dir experiments/my_exp/data/ckpt \
    --resume \
    --ckpt_shard_size 25 \
    --ckpt_flush_secs 60 \
    ...  # 其他原有参数
```

**参数说明：**
- `--ckpt_dir`：checkpoint 目录路径（启用 checkpointing）
- `--resume`：如果 checkpoint 已存在，则继续收集；如果不存在，则创建新的 checkpoint
- `--ckpt_shard_size`：每个 shard 文件包含的轨迹数量（默认 25）
- `--ckpt_flush_secs`：即使缓冲区未满，也每隔多少秒刷新一次 shard（默认 60 秒）

### 2. 查看收集进度

使用 `routes_ckpt_tools.py` 查看 checkpoint 信息：

```bash
python eval/routes_ckpt_tools.py info --ckpt_dir experiments/my_exp/data/ckpt
```

这会显示：
- 已收集的轨迹数量 / 目标数量
- 当前种子（current_seed）
- 已尝试的种子数量
- 已写入的 shard 文件数量
- 配置信息（环境、模型、参数等）

### 3. 导出部分数据

即使还没收集完，也可以将已收集的数据导出为标准 `routes.npz`：

```bash
python eval/routes_ckpt_tools.py build \
    --ckpt_dir experiments/my_exp/data/ckpt \
    --out_npz routes_partial.npz
```

导出的文件可以直接用于 `analysis/pkd_cycle_sampler.py`：

```bash
python analysis/pkd_cycle_sampler.py \
    --routes_npz routes_partial.npz \
    --model_ckpt ... \
    --out_npz pkd_cycles.npz \
    ...
```

### 4. 断点续跑

如果程序中断（手动退出、服务器故障等），只需重新运行相同的命令（包含 `--ckpt_dir` 和 `--resume`），程序会自动：

1. 检测到已存在的 checkpoint
2. 验证配置是否匹配（环境、模型、参数等）
3. 从上次的 `current_seed` 继续收集
4. 不会重复已收集的种子

**注意**：如果配置参数（如 `--env_name`、`--model_ckpt`、`--max_steps` 等）与 checkpoint 中保存的不一致，程序会报错并拒绝继续，除非使用 `--resume_force`（不推荐，可能导致数据不一致）。

## 工作原理

### 存储格式（ckpt_shards_v1）

Checkpoint 目录结构：
```
experiments/my_exp/data/ckpt/
├── manifest.json          # 元数据和进度信息
└── shards/
    ├── shard_000000.npz   # 第 1 个 shard（包含前 25 条轨迹）
    ├── shard_000001.npz   # 第 2 个 shard（包含接下来 25 条轨迹）
    └── ...
```

每个 shard 文件是一个标准的 `routes.npz` 格式，包含该 shard 内的所有轨迹数据。

### 刷新策略

程序会在以下情况刷新 shard（将缓冲区写入磁盘）：

1. **数量触发**：缓冲区中的轨迹数量达到 `--ckpt_shard_size`
2. **时间触发**：距离上次刷新已过去 `--ckpt_flush_secs` 秒

**崩溃安全性**：
- Shard 文件使用原子写入（先写 `.tmp` 文件，再 `os.replace`）
- `manifest.json` 只在 shard 安全写入后才更新
- 程序收到 SIGINT/SIGTERM 时会先刷新缓冲区再退出

**数据丢失风险**：
- 最坏情况下，可能丢失最近一个 shard 内的数据（最多 `ckpt_shard_size - 1` 条轨迹）
- 如果程序在刷新间隔内崩溃，可能丢失最近 `ckpt_flush_secs` 内收集的数据

### 最终输出

收集完成后，程序会自动将所有 shard 合并成最终的 `--out_npz` 文件，格式与不使用 checkpointing 时完全相同。

## 完整示例

### 示例 1：收集 10000 条轨迹（带 checkpointing）

```bash
#!/bin/bash
EXP_NAME="my_collection"
CKPT_DIR="experiments/${EXP_NAME}/data/ckpt"
ROUTES_NPZ="experiments/${EXP_NAME}/data/routes.npz"

python eval/collect_meta_routes.py \
    --env_name coinrun \
    --model_ckpt /path/to/model.tar \
    --out_npz "${ROUTES_NPZ}" \
    --num_tasks 10000 \
    --num_processes 128 \
    --device cuda:0 \
    --ckpt_dir "${CKPT_DIR}" \
    --resume \
    --ckpt_shard_size 25 \
    --ckpt_flush_secs 60
```

### 示例 2：查看进度并导出部分数据

```bash
# 查看进度
python eval/routes_ckpt_tools.py info --ckpt_dir experiments/my_collection/data/ckpt

# 如果已收集了 5000 条，想先用这 5000 条跑分析
python eval/routes_ckpt_tools.py build \
    --ckpt_dir experiments/my_collection/data/ckpt \
    --out_npz experiments/my_collection/data/routes_5k.npz

# 用部分数据跑 PKD
python analysis/pkd_cycle_sampler.py \
    --routes_npz experiments/my_collection/data/routes_5k.npz \
    --model_ckpt /path/to/model.tar \
    --out_npz experiments/my_collection/analysis/pkd_cycles_5k.npz \
    ...
```

### 示例 3：断点续跑

```bash
# 第一次运行（收集到 3000 条时手动退出或崩溃）
python eval/collect_meta_routes.py \
    --ckpt_dir experiments/my_collection/data/ckpt \
    --resume \
    --num_tasks 10000 \
    ...

# 第二次运行（自动从第 3000 条继续，收集剩余的 7000 条）
# 使用完全相同的命令即可
python eval/collect_meta_routes.py \
    --ckpt_dir experiments/my_collection/data/ckpt \
    --resume \
    --num_tasks 10000 \
    ...
```

## 常见问题

### Q: 如果不使用 `--ckpt_dir`，行为会改变吗？

**A:** 不会。如果不提供 `--ckpt_dir`，程序行为与之前完全相同：所有数据保存在内存中，最后一次性写入 `--out_npz`。

### Q: 可以修改配置参数后继续收集吗？

**A:** 默认情况下不可以。如果配置参数（环境、模型、max_steps 等）与 checkpoint 中保存的不一致，程序会拒绝继续，避免数据不一致。如果确实需要修改配置，可以使用 `--resume_force`，但**不推荐**，可能导致数据质量问题。

### Q: checkpoint 目录可以删除吗？

**A:** 可以。收集完成并确认最终 `routes.npz` 文件正确后，可以删除 checkpoint 目录以节省空间。checkpoint 只是临时存储，最终输出是 `--out_npz` 文件。

### Q: 可以同时运行多个收集任务使用同一个 checkpoint 目录吗？

**A:** **不可以**。多个进程同时写入同一个 checkpoint 目录会导致数据损坏。每个收集任务必须使用独立的 checkpoint 目录。

### Q: 如何选择合适的 `--ckpt_shard_size`？

**A:** 
- **较小值（10-25）**：更频繁的刷新，崩溃时丢失数据更少，但会产生更多 shard 文件
- **较大值（50-100）**：更少的文件，但崩溃时可能丢失更多数据

建议：对于长时间运行的任务（如收集 10000 条），使用默认值 25 即可。

### Q: 导出的部分数据可以用于所有分析脚本吗？

**A:** 可以。导出的 `routes.npz` 格式与完整收集的格式完全相同，可以用于：
- `analysis/pkd_cycle_sampler.py`
- `analysis/cca_alignment.py`
- `analysis/trajectory_stats.py`
- 其他所有使用 `routes.npz` 的脚本

## 测试脚本

仓库中提供了完整的测试脚本，演示整个流程：

1. **收集测试**：`run_collect_test_coinrun_ckpt.sh`
   - 收集少量轨迹（20 条）并启用 checkpointing
   - 验证 checkpoint 功能

2. **分析测试**：`run_analysis_test_coinrun_ckpt.sh`
   - 从 checkpoint 导出部分数据
   - 在导出的数据上运行 PKD cycle sampler
   - 验证整个流程

运行测试：
```bash
# 1. 运行收集测试
./run_collect_test_coinrun_ckpt.sh

# 2. 运行分析测试
./run_analysis_test_coinrun_ckpt.sh
```

## 技术细节

### Manifest 文件格式

`manifest.json` 包含以下信息：
- 存储格式版本（`storage_format`）
- 创建/更新时间戳
- 收集进度（已收集数量、当前种子、已尝试种子数）
- 配置参数（用于 resume 时验证）
- Shard 文件数量

### Shard 文件格式

每个 shard 文件是一个标准的 `npz` 文件，包含：
- `routes_seed`、`routes_obs`、`routes_actions` 等所有标准字段
- 格式与最终输出的 `routes.npz` 完全相同

### 合并过程

`build_routes_npz_from_ckpt` 函数会：
1. 扫描所有 shard 文件（按文件名排序）
2. 逐个加载并合并所有字段
3. 生成最终的 `routes.npz`，包含所有轨迹的合并数据
4. 添加 `meta` 字典，包含收集配置信息

## 总结

Checkpointing 功能提供了：
- ✅ **崩溃安全性**：定期保存进度，避免数据丢失
- ✅ **断点续跑**：自动从上次进度继续，不重复工作
- ✅ **灵活分析**：可以随时导出部分数据进行初步分析
- ✅ **向后兼容**：不影响现有脚本，完全可选

建议在收集大量轨迹（如 10000+）时启用此功能，特别是当收集时间较长或服务器可能不稳定时。

