# EDE Conda 环境重建指南

本文档提供在新机器上重建 `ede` conda 环境的详细步骤。

## 📋 环境概要

- **环境名称**: ede
- **Python 版本**: 3.10.18
- **主要框架**: PyTorch 2.5.1 (CUDA 12.1)
- **导出日期**: 2025-12-24

## 🔧 核心依赖包

### 深度学习框架
- **PyTorch**: 2.5.1+cu121
- **TorchVision**: 0.20.1+cu121
- **TorchAudio**: 2.5.1+cu121
- **Kornia**: 0.8.1 (计算机视觉库)

### 强化学习相关
- **Gym**: 0.26.2
- **Gym3**: 0.3.3
- **Procgen**: 0.10.7
- **Gym-MiniGrid**: 1.2.2
- **Baselines**: 0.1.6
- **CircRL**: 1.0.0

### 科学计算
- **NumPy**: 1.26.4
- **SciPy**: 1.10.0
- **Pandas**: 2.3.3
- **Scikit-learn**: 1.7.2
- **Statsmodels**: 0.14.6

### 可视化
- **Matplotlib**: 3.3.2
- **Seaborn**: 0.13.2
- **Plotly**: 6.5.0
- **Bokeh**: 3.8.1
- **HoloViews**: 1.22.1

### Jupyter 生态
- **JupyterLab**: 4.1.6
- **Notebook**: 7.1.3
- **IPython**: 8.37.0

### 其他工具
- **Wandb**: 0.22.1 (实验追踪)
- **GPUStat**: 1.1.1 (GPU监控)
- **OpenCV**: 4.12.0.88
- **Captum**: 0.8.0 (模型解释)

## 🚀 重建方法

### 方法 1: 使用 environment.yml（推荐）

这是最简单、最可靠的方法，会完整复制整个环境。

```bash
# 1. 将 ede_environment.yml 文件复制到新机器

# 2. 创建新环境
conda env create -f ede_environment.yml

# 3. 激活环境
conda activate ede

# 4. 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 方法 2: 使用 requirements.txt

如果需要更灵活的安装方式：

```bash
# 1. 创建基础环境
conda create -n ede python=3.10.18

# 2. 激活环境
conda activate ede

# 3. 使用 conda 安装（可选，如果 requirements.txt 中有 conda 包）
conda install --file ede_requirements.txt

# 注意：这个方法可能遇到 pip 包的问题，推荐使用方法1
```

### 方法 3: 手动安装关键包（适用于有定制需求的情况）

```bash
# 1. 创建基础环境
conda create -n ede python=3.10.18 -y
conda activate ede

# 2. 安装 PyTorch (根据您的CUDA版本调整)
# CUDA 12.1 版本：
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装强化学习库
pip install gym==0.26.2 gym3==0.3.3 procgen==0.10.7 gym-minigrid==1.2.2

# 4. 安装科学计算库
pip install numpy==1.26.4 scipy==1.10.0 pandas==2.3.3 scikit-learn==1.7.2

# 5. 安装可视化库
pip install matplotlib==3.3.2 seaborn==0.13.2 plotly==6.5.0

# 6. 安装 Jupyter
pip install jupyterlab==4.1.6 notebook==7.1.3

# 7. 安装其他依赖
pip install wandb==0.22.1 opencv-python==4.12.0.88 kornia==0.8.1
```

## ⚠️ 重要注意事项

### CUDA 兼容性
此环境使用 **CUDA 12.1**。确保您的机器上：
- 安装了 NVIDIA 驱动（建议 >= 525.x）
- GPU 支持 CUDA 12.1
- 如果 CUDA 版本不同，需要重新安装对应版本的 PyTorch

### 镜像源配置
原环境使用了清华镜像源。如果需要，可以配置：

```bash
# 查看当前 channels
conda config --show channels

# 添加清华镜像（可选）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels conda-forge
```

### 平台兼容性
- 此环境导出自 **Linux (linux-64)** 平台
- 在 Windows 或 macOS 上可能需要调整某些包的版本
- 某些包（如 procgen）在不同平台上可能有兼容性问题

### 自定义包
注意到环境中有一个开发包：
- `procgen-tools=0.1.1=dev_0` 

这可能是本地安装的开发版本，需要单独处理。

## ✅ 验证安装

安装完成后，运行以下命令验证：

```bash
conda activate ede

# 检查 Python 版本
python --version

# 检查 PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 检查关键包
python -c "import gym, procgen, numpy, pandas, matplotlib, wandb; print('所有关键包导入成功')"

# 查看已安装包
conda list

# 检查环境信息
conda info
```

## 📦 文件说明

此目录包含以下文件：

1. **ede_environment.yml** - 完整的 conda 环境配置文件（推荐使用）
2. **ede_requirements.txt** - conda format 的包列表
3. **EDE_ENVIRONMENT_SETUP.md** - 本说明文档

## 🐛 常见问题

### 问题1: CUDA 版本不匹配
**解决方案**: 根据您的 CUDA 版本重新安装 PyTorch
```bash
# 查看 CUDA 版本
nvidia-smi

# 访问 https://pytorch.org 选择对应版本
```

### 问题2: 某些包安装失败
**解决方案**: 尝试单独安装失败的包
```bash
pip install <package-name>==<version>
```

### 问题3: 环境创建时间过长
**解决方案**: 使用 mamba 替代 conda
```bash
conda install mamba -n base -c conda-forge
mamba env create -f ede_environment.yml
```

## 📞 技术支持

如有问题，请检查：
- Conda 版本是否 >= 4.10
- 网络连接是否正常
- 磁盘空间是否充足（建议 > 10GB）

---

**生成时间**: 2025-12-24  
**源环境路径**: /root/miniconda3/envs/ede  
**Python 版本**: 3.10.18
