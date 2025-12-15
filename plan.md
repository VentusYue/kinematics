# Procgen Meta-RL（RL²）复现 Figure 5：Behavior–Neural State Alignment（PKD + Ridge + CCA）计划

> 目标：在 `VentusYue/kinematics` 代码库（Procgen + recurrent PPO + meta-RL/RL²）上，复现论文 *meta learning maze navigation* 的 **behavior–neural state alignment**（Figure 5）流程与图形输出。
>
> 关键词闭环：**明确分析输入/输出 → eval 采样（含 x,y）→ PKD 极限环采样 → 行为 ridge 嵌入 → 神经状态聚合 → CCA 对齐 → Figure5 可视化**。

---

## 0. 你现在已有的资产（Inputs）

### 0.1 模型与训练代码（已存在）
- **Environment**: `conda activate ede`
- **Model Checkpoint**: `/root/logs/ppo/meta-rl-maze-dense-long-n1280meta-40gpu1/model.tar`
- 训练脚本：`train_ppo_recurrent.py`（RL² 风格 meta-RL）。

### 0.2 现状判断：rnn_traces 能不能直接用？
**结论：已弃用 rnn_traces，改用新写的收集脚本。**
原因：
1) `rnn_traces` 是训练截断片段，缺乏完整成功路线。
2) 缺 (x,y) 坐标。
3) **状态更新**：已实现 `eval/collect_meta_routes.py`，专门用于采集含 (x,y) 的完整轨迹。

### 0.3 环境与依赖配置（已验证）
1.  **Procgen Tools**:
    - 已安装并打补丁（`analysis/procgen_xy.py` 中处理了 `ProcgenGym3Env` 导入问题）。
    - 验证：`get_xy_from_venv` 可正确提取 agent 坐标。
2.  **Dependencies**:
    - `pip install bidict pandas plotly statsmodels prettytable` (已安装)
    - `numpy` bool patch (已在脚本中处理)

---

## 1. Figure5 复现的“数据闭环”定义（Inputs → Outputs）

### 1.1 最终输出（Outputs）
1) `analysis_out/figure5_alignment.png`
   - 左：神经环（或环的代表点）在 canonical space (CM0, CM1) 的散点
   - 右：行为 ridge 嵌入在同一 canonical space (CM0, CM1) 的散点
   - 颜色：路径长度/episode length
2) `analysis_out/cca_lollipop.png`：前 K 个 canonical correlation 系数
3) 可选 debug：
   - `analysis_out/debug_ridge_examples.png`
   - `analysis_out/debug_pkd_cycle_pca.png`

### 1.2 中间产物
- `analysis_data/routes_meta_eval.npz`：**已通过测试运行生成** (eval 采样得到的成功路线库)
- `analysis_data/pkd_cycles.npz`：每条成功路线 → PKD 诱导极限环 → hidden-cycle 库
- `analysis_data/cca_inputs.npz`：最终用于 CCA 的 X/Y 矩阵

---

## 2. 依赖与关键技术选择

### 2.1 关键依赖：提取 Procgen Maze 的 (x,y)
**状态：已完成 (Implemented & Verified)**
- 脚本：`analysis/procgen_xy.py`
- 方案：通过 `procgen-tools` 解析 C++ state bytes。
- 修复：解决了 `gym` vs `gymnasium` 及 `ProcgenGym3Env` 缺失的兼容性问题。

---

## 3. 代码改造总览（在 kinematics repo 上落地）

### 新增文件
- `analysis/procgen_xy.py`：**(已完成)** 坐标提取接口。
- `eval/collect_meta_routes.py`：**(已完成)** 路线采集脚本。
- `analysis/ridge_embedding.py`：(待办) ridge image / ridge vector。
- `analysis/pkd_cycle_sampler.py`：(待办) PKD 诱导极限环采样。
- `analysis/cca_alignment.py`：(待办) CCA 分析与绘图。

---

## 4. Eval：在现有模型上怎么跑、跑多少、改哪些、得到什么

### 4.1 Eval 配置建议
- `num_tasks`：**200–500**
- `trials_per_task`：5
- policy：**建议使用 `deterministic=0` (Stochastic)**。
  - *实验发现*：Deterministic policy 在部分 seed 上容易卡死或不成功；Stochastic policy 成功率更高，生成的轨迹更多样，适合分析。
- `num_processes`：1 (串行采集以保证数据完整性)

### 4.2 Eval 脚本实现 (`eval/collect_meta_routes.py`)
**状态：已完成并测试通过**
- 功能：加载 checkpoint，运行 RL² 流程，记录 obs/action/hidden/xy。
- 输出：`.npz` 文件，包含 object array 格式的变长轨迹。

### 4.3 Eval 得到什么（routes_meta_eval.npz）
字段（已验证）：
- `routes_obs`: (N_routes, T, 3, 64, 64)
- `routes_xy`: (N_routes, T, 2)
- `routes_actions`: (N_routes, T)
- `routes_hiddens`: (N_routes, T, H)
- `routes_seed`, `routes_trial_id`, `routes_ep_len`, `routes_return`, `routes_success`

---

## 5. PKD 极限环采样（从 route → neural cycle）

### 5.1 PKD 输入是什么？
对每条选中的 route：
- 周期驱动序列 `O = (o_0, o_1, ..., o_{L-1})`：来自 eval 中该 trial 的 obs 序列
- 目标动作序列 `A* = (a*_0, ..., a*_{L-1})`：同一条 trial 的 actions（用于 ACF）

### 5.2 PKD 采样怎么做？
`analysis/pkd_cycle_sampler.py`：
1) 冻结模型参数（`actor_critic.eval(); torch.no_grad()`）。
2) 采样多个初始 hidden `h0`（M 次，建议 10–50）：
   - 方式 A：`h0 ~ N(0, 1)`  
   - 方式 B（更稳）：从 eval 里收集 “task start hidden” 的统计量，用其均值/方差采样
3) 对每个 h0：
   - 将 `O` 重复喂给网络 `P` 个 period（例如 P=10，前 8 个当 warmup）
   - 取最后 1–2 个 period 的 hidden 轨迹，检查收敛（上一周期与下一周期平均 L2 差 < tol）
4) **ACF（Action Consistency Filter）**
   - 用 deterministic action（mode）得到 PKD 下的动作序列 `A_pkd`
   - 计算 `match_ratio = mean(A_pkd == A*)`
   - `match_ratio >= 0.95` 才接受为“on-strategy cycle”
5) 保存：
   - `hidden_cycle`: (L, H)
   - `route_id`: 指向行为 ridge
   - 可选：`pkd_actions`, `match_ratio`, `converge_score`

输出：`analysis_data/pkd_cycles.npz`

---

## 6. 行为嵌入：x,y → ridge image → ridge vector

### 6.1 (x,y) 处理
- 来源：`analysis/procgen_xy.py` (已验证)
- 预处理：
  - 归一化：`x_tile = x / grid_step`
  - 平移对齐：`start_pos -> (10, 10)`

### 6.2 ridge embedding
- 待实现：将轨迹转换为 21x21 的 radiance field。

---

## 7. 神经聚合与 CCA 对齐（Figure 5）

### 7.1 CCA 输入矩阵 X/Y 怎么构造？
推荐（更贴近论文的 “states on/near the cycle”）：
- 对每条 cycle（hidden_cycle, ridge_vec）：
  - 取 cycle 上的若干相位点 `t`（例如全取 or 每隔 k 取）
  - X：堆叠 `h_t`（shape: n_samples × H）
  - Y：堆叠对应的 `ridge_vec`（shape: n_samples × 441），同一 cycle 会重复同一 ridge_vec

然后跑线性 CCA（你的 `canoncorr` 即可）。

### 7.2 Figure 5 的“每个 cycle 一个点”怎么得到？
- 神经侧：对同一个 cycle 的 U（canonical scores）在时间维上取均值 → `U_cycle_mean` (d,)
- 行为侧：同一 cycle 的 ridge_vec 对应的 V 是常量（或重复取均值）→ `V_cycle` (d,)

最后画两个散点：
- 左图：`U_cycle_mean[:, 0]` vs `U_cycle_mean[:, 1]`
- 右图：`V_cycle[:, 0]` vs `V_cycle[:, 1]`
颜色用 `route_ep_len` 或 `path_length`（你 paper 里用哪个就跟哪个一致）。

---

## 8. 最小可复现实验命令（Updated）

### 8.1 收集 routes（eval）
```bash
# 激活环境
conda activate ede

# 执行收集
python eval/collect_meta_routes.py \
  --env_name maze \
  --model_ckpt /root/logs/ppo/meta-rl-maze-dense-long-n1280meta-40gpu1/model.tar \
  --num_tasks 200 \
  --trials_per_task 5 \
  --deterministic 0 \
  --num_processes 1 \
  --hidden_size 256 \
  --arch large \
  --out_npz analysis_data/routes_meta_eval.npz
```

### 8.2 PKD 采样 cycles
```bash
python analysis/pkd_cycle_sampler.py \
  --model_ckpt /root/logs/ppo/meta-rl-maze-dense-long-n1280meta-40gpu1/model.tar \
  --routes_npz analysis_data/routes_meta_eval.npz \
  --num_h0 20 \
  --warmup_periods 8 \
  --sample_periods 2 \
  --ac_match_thresh 0.95 \
  --out_npz analysis_data/pkd_cycles.npz
```

### 8.3 CCA 对齐 + Figure5
```bash
python analysis/cca_alignment.py \
  --cycles_npz analysis_data/pkd_cycles.npz \
  --out_dir analysis_out \
  --num_modes 10
```

---

## 9. 里程碑（Status Update）

1.  **(Done)** `(x,y)` 抽取接口 (`analysis/procgen_xy.py`)：**已完成并验证**。
2.  **(Done)** Eval 采集脚本 (`eval/collect_meta_routes.py`)：**已完成**，成功生成含坐标的 `npz`。
3.  **(Done)** PKD 采样 (`analysis/pkd_cycle_sampler.py`)：**已完成并验证**。
4.  **(Done)** Ridge Embedding (`analysis/ridge_embedding.py`)：**已完成**。
5.  **(Done)** CCA & Visualization (`analysis/cca_alignment.py`)：**已完成并验证**。

---

## 10. 交付标准
- 执行上述 3 条命令可生成 `figure5_alignment.png`。
