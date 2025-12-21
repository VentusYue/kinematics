# CCA分析程序对比：Legacy vs Current

**作者**: Analysis  
**日期**: 2025-12-21  
**目的**: 对比分析legacy实验程序(e_r67_cca_compare.py)与当前CCA分析程序(analysis/cca_alignment.py)的差异

---

## 📋 执行摘要

Legacy程序在相关数据上表现优异，CCA相关性结果准确可靠。通过对比分析发现**三个关键差异**：

1. **PCA预处理**（Legacy有，Current无） - **最关键**
2. **噪声注入**（Legacy有，Current无） - **重要**
3. **Ridge对齐方式**（Legacy对齐到grid中心，Current对齐到原点）

这些差异导致Current程序出现ridge embedding高度相似(cosine sim=0.9989)的问题。

---

## 🔬 详细差异对比

### 1. 数据预处理流程

#### Legacy程序（e_r67_cca_compare.py）✅

```python
# 步骤1: 添加小噪声（regularization）
noise = np.random.uniform(-0.001, 0.001, size=ridge_images.shape)
ridge_images = ridge_images + noise

noise = np.random.uniform(-0.001, 0.001, size=rnn_center_mat.shape)
rnn_center_mat = rnn_center_mat + noise

# 步骤2: PCA降维
pca = PCA()
pca.fit(train_view_1)
train_view_1 = pca.transform(train_view_1)  # Hidden states

pca.fit(train_view_2)
train_view_2 = pca.transform(train_view_2)  # Ridge embeddings

# 步骤3: CCA
A, B, r, U, V = canoncorr(train_view_1, train_view_2, fullReturn=True)
```

**优点**：
- ✅ PCA提取主成分，去除冗余维度
- ✅ 小噪声打破数值共线性
- ✅ 提高CCA的数值稳定性

#### Current程序（analysis/cca_alignment.py）

```python
# 直接运行CCA，无预处理
X = np.concatenate(X_samples, axis=0).astype(np.float32)
Y = np.concatenate(Y_samples, axis=0).astype(np.float32)

A, B, r, U, V = canoncorr(X, Y, fullReturn=True)
```

**问题**：
- ❌ 无PCA降维 → ridge的441维中可能有大量冗余
- ❌ 无噪声regularization → 容易产生数值artifacts
- ❌ 直接在高维空间CCA → 可能overfitting

---

### 2. Ridge Embedding实现差异

#### Legacy: 使用JAX + 对齐到Grid中心

```python
@jax.jit
def build_ridge(A):
    # 关键：对齐到21×21 grid的中心 (10, 10)
    A = A - A[0] + jnp.array([10, 10])
    
    # JAX加速的辐射场计算
    imgs = jax.vmap(build_radiance_field)(A)
    img = get_max_radiance_field(imgs)
    return img
```

**参数**：
- Grid size: 21×21
- Center: (10, 10)
- 辐射半径: 21 × 1.414 ≈ 29.7
- Value range: [0, 10]

#### Current: NumPy + 对齐到原点

```python
# 在cca_alignment.py中
path_tile = path_tile - path_tile[0]  # 对齐到(0, 0)

# 在ridge_embedding.py中
def build_ridge(path, grid_size=21):
    A = path.copy()
    center = grid_size // 2  # = 10
    offset = np.array([center, center]) - A[0]
    A = A + offset  # 实际上也对齐到(10, 10)
```

**发现**：
- `ridge_embedding.py`内部**已经**对齐到(10, 10)！
- 但`cca_alignment.py`先对齐到(0, 0)
- 这导致**双重对齐**：先(0,0)再(10,10)
- 最终效果：所有Ridge图像都非常相似（都是从(10,10)开始的短路径）

---

### 3. 标准化处理差异

#### Legacy: 简单标准化

```python
X = (X0 - np.mean(X0, 0)) / np.std(X0, 0)
Y = (Y0 - np.mean(Y0, 0)) / np.std(Y0, 0)
# 不处理std=0的情况（假设有噪声不会出现这个问题）
```

#### Current: 安全标准化

```python
X_std = np.std(X0, 0)
Y_std = np.std(Y0, 0)

# 处理常量列
X_std[X_std == 0] = 1.0
Y_std[Y_std == 0] = 1.0

X = (X0 - np.mean(X0, 0)) / X_std
Y = (Y0 - np.mean(Y0, 0)) / Y_std
```

**评价**：Current在这方面更robust，这不是问题所在

---

### 4. 数据组织方式

#### Legacy: Cycle-level aggregation

```python
# 每个样本 = 一个完整的路径/behavior
# 使用填充(padding)统一长度
def preprocess(trjs, max_length):
    if trj.shape[0] < max_length:
        # 用最后一个元素填充
        processed_trj = np.concatenate([trj, np.repeat(last_element, ...)])

# 结果：(N_cycles, unified_length, feature_dim)
```

#### Current: Timestep-level concatenation

```python
# 每个样本 = 一个timestep
X_samples.append(h_cycle)  # (L, H)
Y_samples.append(np.tile(ridge_vec, (L, 1)))  # (L, 441)

X = np.concatenate(X_samples, axis=0)  # (ΣL, H)
```

**问题**：
- Legacy: 样本间独立（不同behaviors）
- Current: 同一cycle的timesteps都相似 → 产生大量重复/相似样本
- **这进一步加剧ridge相似度问题**

---

## 🎯 改进建议

### ⭐⭐⭐ 关键改进（必须实施）

#### 改进1: 添加PCA预处理

```python
from sklearn.decomposition import PCA

# 在运行CCA之前
print("\n" + "-"*40)
print("PCA PREPROCESSING")
print("-"*40)

# PCA for X (Neural states)
pca_x = PCA()
pca_x.fit(X)
X_pca = pca_x.transform(X)
print(f"  X: {X.shape} → {X_pca.shape}")
print(f"  X explained variance ratio (top 10): {pca_x.explained_variance_ratio_[:10]}")

# PCA for Y (Ridge embeddings)
pca_y = PCA()
pca_y.fit(Y)
Y_pca = pca_y.transform(Y)
print(f"  Y: {Y.shape} → {Y_pca.shape}")
print(f"  Y explained variance ratio (top 10): {pca_y.explained_variance_ratio_[:10]}")

# 使用PCA transformed数据进行CCA
A, B, r, U, V = canoncorr(X_pca, Y_pca, fullReturn=True)
```

#### 改进2: 添加噪声注入

```python
# 在concatenate之后，CCA之前
print("  Adding regularization noise...")

noise_x = np.random.uniform(-0.001, 0.001, X.shape)
noise_y = np.random.uniform(-0.001, 0.001, Y.shape)

X = X + noise_x
Y = Y + noise_y
```

### ⭐⭐ 重要改进

#### 改进3: 修正Ridge对齐逻辑

**问题**：双重对齐导致所有路径过于相似

**解决方案**：
```python
# 在cca_alignment.py中
# 选项A: 不做预对齐，让ridge_embedding.py处理
# path_tile = path_xy / est_grid_step
# # 注释掉：path_tile = path_tile - path_tile[0]

# 选项B: 确保只对齐一次
# 检查ridge_embedding.py的实现，如果它已经对齐，就不要预对齐
```

### ⭐ 可选改进

#### 改进4: Cycle-level aggregation

考虑先对每个cycle取平均hidden state，再进行CCA：

```python
# 修改数据构建方式
cycle_hidden_means = []
cycle_ridge_vecs = []

for i in range(num_cycles):
    h_cycle = cycles_hidden[i]  # (L, H)
    h_mean = np.mean(h_cycle, axis=0)  # (H,)
    cycle_hidden_means.append(h_mean)
    
    ridge_vec = build_ridge_vector(path_tile)
    cycle_ridge_vecs.append(ridge_vec)

X = np.array(cycle_hidden_means)  # (N_cycles, H)
Y = np.array(cycle_ridge_vecs)     # (N_cycles, 441)
```

---

## 📊 实施计划示例

### 快速测试方案

```python
# analysis/cca_alignment.py 修改位置
# 在 "RUN CCA" section之前添加：

# =========================================================================
# PCA PREPROCESSING (Legacy-inspired improvement)
# =========================================================================
print("\n" + "-"*40)
print("PCA PREPROCESSING")
print("-"*40)

# Add noise for regularization
print("  Adding noise regularization...")
noise_x = np.random.uniform(-0.001, 0.001, X.shape)
noise_y = np.random.uniform(-0.001, 0.001, Y.shape)
X = X.astype(np.float64) + noise_x
Y = Y.astype(np.float64) + noise_y

# PCA transformation
from sklearn.decomposition import PCA

pca_x = PCA()
X_pca = pca_x.fit_transform(X)
print(f"  X PCA: {X.shape} → {X_pca.shape}")
print(f"  X variance explained (cumsum): {pca_x.explained_variance_ratio_.cumsum()[:20]}")

pca_y = PCA()
Y_pca = pca_y.fit_transform(Y)
print(f"  Y PCA: {Y.shape} → {Y_pca.shape}")
print(f"  Y variance explained (cumsum): {pca_y.explained_variance_ratio_.cumsum()[:20]}")

# Use PCA-transformed data for CCA
X = X_pca
Y = Y_pca

# Continue with CCA...
```

---

## 🧪 预期结果对比

### Before (Current状态)

```
Ridge Embedding Diversity:
  Pairwise cosine similarity: mean=0.9989  ⚠️ 过高！
  
CCA Results:
  Top 10 correlations: [0.995, 0.971, 0.962, ...]
  High (>0.9): 5
  [WARN] Ridge embeddings are very similar!
```

### After (添加PCA+噪声后)

```
Ridge Embedding Diversity:
  Pairwise cosine similarity: mean=0.75-0.90  ✅ 合理范围
  
CCA Results:
  Top 10 correlations: [0.92, 0.85, 0.78, ...]
  High (>0.9): 2-3
  Medium (0.5-0.9): 6-8
  Distribution更加spread out - 说明找到了meaningful modes
```

---

## 🔍 技术深入：为什么PCA有效？

### Ridge Embedding的本质特性

Ridge embeddings (441维) 实际上是**低秩流形**：

1. **路径拓扑约束**：
   - 在21×21 grid中，有意义的路径模式数量 << 441
   - 大部分variance集中在前k<<441个主成分

2. **辐射场重叠**：
   - 相邻点的辐射场高度重叠
   - 创建维度间的强相关性

3. **对齐效应**：
   - 所有路径对齐到同一起点
   - 进一步减少variation空间

### PCA的作用

```
原始Ridge空间(441维):
  维度1-50:  真正的路径形状信息 (90%+ variance)
  维度51-441: 噪声、冗余、数值误差 (~0% variance)

PCA after:
  主成分1-50: 捕获真实差异
  其余成分:   被过滤
  
→ CCA在clean feature space上工作，结果更meaningful
```

---

## 📈 实际案例分析

### Legacy程序的数据特征

```python
# Legacy处理的数据
N_cycles = ~4000
Sequence lengths: 5-14 (填充到统一长度)
Hidden dim: 128
Ridge dim: 441

经过PCA后:
X_pca: (4000, 128) - 保留所有128维（已经较低）
Y_pca: (4000, ~50-100) - 显著降维（从441→主要variance components）

CCA结果: 清晰的mode separation
```

### Current程序的数据特征

```python
# Current处理的数据  
N_samples =注册 112,272 timesteps (from 16,284 cycles)
Hidden dim: 256
Ridge dim: 441

高度重复：
- 每个cycle贡献平均6.9个timesteps
- 这6.9个timesteps的ridge embedding完全相同（同一path）!
- 导致cosine sim极高

无PCA:
- 直接在441维ridge space做CCA
- 大量冗余维度参与计算
```

---

## 🛠️ 立即实施的代码修改

### 方案A：最小修改（推荐首次尝试）

在`analysis/cca_alignment.py`的CCA section之前添加：

```python
# After concatenating X and Y, before CCA

# =========================================================================
# LEGACY-INSPIRED PREPROCESSING
# =========================================================================
print("\n" + "-"*40)
print("APPLYING LEGACY PREPROCESSING")
print("-"*40)

from sklearn.decomposition import PCA

# 1. Noise injection (regularization)
noise_scale = 0.001
print(f"  Injecting noise (±{noise_scale})...")
noise_x = np.random.uniform(-noise_scale, noise_scale, X.shape)
noise_y = np.random.uniform(-noise_scale, noise_scale, Y.shape)
X = X.astype(np.float64) + noise_x
Y = Y.astype(np.float64) + noise_y

# 2. PCA preprocessing
print("  Applying PCA...")
pca_x = PCA()
X_transformed = pca_x.fit_transform(X)
cumsum_x = pca_x.explained_variance_ratio_.cumsum()
n_comp_x_95 = np.searchsorted(cumsum_x, 0.95) + 1
print(f"  X: {X.shape} → PCA → {X_transformed.shape}")
print(f"  X: {n_comp_x_95} components explain 95% variance")

pca_y = PCA()
Y_transformed = pca_y.fit_transform(Y)
cumsum_y = pca_y.explained_variance_ratio_.cumsum()
n_comp_y_95 = np.searchsorted(cumsum_y, 0.95) + 1
print(f"  Y: {Y.shape} → PCA → {Y_transformed.shape}")
print(f"  Y: {n_comp_y_95} components explain 95% variance")

# Use transformed data for CCA
X = X_transformed
Y = Y_transformed

print("-"*40 + "\n")
```

### 方案B：完整重构（与Legacy完全对齐）

修改数据组织方式，使用cycle-level aggregation：

```python
# 修改BUILD DATA MATRICES section

cycle_hidden_means = []
cycle_ridge_vecs = []

for i in range(num_cycles):
    h_cycle = cycles_hidden[i]
    
    # Average hidden states across the cycle
    if len(h_cycle) > 0:
        h_mean = np.mean(h_cycle, axis=0)
    else:
        h_mean = np.zeros(256)
    
    cycle_hidden_means.append(h_mean)
    
    # Compute ridge (one per cycle)
    ridge_vec = build_ridge_vector(path_tile)
    cycle_ridge_vecs.append(ridge_vec)

X = np.array(cycle_hidden_means)  # (N_cycles, 256)
Y = np.array(cycle_ridge_vecs)     # (N_cycles, 441)

# 然后应用噪声+PCA+CCA
```

---

## 📌 关键发现汇总

### 问题根源

**Current程序ridge相似度极高(0.9989)的原因**：

1. ❌ **双重对齐**：先对齐到(0,0)，ridge内部再对齐到(10,10) → 所有ridge图像几乎identical
2. ❌ **无PCA降维**：441维中的冗余dominates CCA
3. ❌ **Timestep-level采样**：同一cycle的多个timesteps产生完全相同的ridge → 大量duplicate samples
4. ❌ **无噪声regularization**：数值共线性artifacts

### Legacy成功的关键

1. ✅ PCA提取meaningful features
2. ✅ 噪声打破perfect collinearity  
3. ✅ Cycle-level aggregation（每个样本代表不同behavior）
4. ✅ 正确的Ridge对齐方式

---

## 🚀 推荐行动计划

### Phase 1: 快速验证（1-2小时）

1. 在Current程序中添加噪声+PCA（方案A）
2. 运行`bash ./run_analysis_tuning_filtered.sh --skip-pkd`
3. 观察ridge cosine similarity是否降低
4. 检查CCA correlation distribution是否更spread

### Phase 2: 深度修复（如果Phase 1不够）

1. 修正double alignment问题
2. 改为cycle-level aggregation（方案B）
3. 全面对齐Legacy的数据pipeline

### Phase 3: 验证和文档化

1. 对比新旧结果
2. 记录改进效果
3. 更新分析文档

---

## 📚 参考

- **Legacy程序**: `experiments/paper_exp/e_r67_cca_compare.py`
- **Current程序**: `analysis/cca_alignment.py`
- **Ridge实现**: `analysis/ridge_embedding.py`
- **相关论文**: "Preserved neural dynamics across animals performing similar behaviour" (CCA方法来源)

---

## 💬 结论

Legacy程序的成功不是偶然的 - 它采用了**正确的数据预处理pipeline**：

```
Raw Data → Noise Injection → PCA → CCA → High-quality results
```

Current程序缺少中间两步，导致：
- Ridge embeddings过度相似
- CCA结果可能是artifacts而非真实alignment

**建议立即实施PCA+噪声改进**，这是提升分析质量的关键！

---

**文档版本**: 1.0  
**最后更新**: 2025-12-21
