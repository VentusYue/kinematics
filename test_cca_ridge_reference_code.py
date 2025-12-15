"""
测试程序：演示ridge image转换和CCA分析
从 e_r67_cca_compare.py 中提取核心功能（改用NumPy而非JAX）
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd, inv
import logging

# ==================== Ridge Image 相关函数（NumPy版本）====================

def build_radiance_field(p):
    """构建单个点的辐射场"""
    img = np.zeros((21, 21))
    top = 10
    bottom = 0
    effective_radius = 21*1.414
    px, py = p[0], p[1]

    x_coord_map = np.arange(img.shape[0])
    x_coord_map = np.repeat(x_coord_map[:, np.newaxis], img.shape[1], axis=1)

    y_coord_map = np.arange(img.shape[1])
    y_coord_map = np.repeat(y_coord_map[np.newaxis, :], img.shape[0], axis=0)
    
    dist = np.sqrt((x_coord_map - px)**2 + (y_coord_map - py)**2)
    img = np.where(dist < effective_radius,
                   (top - bottom) * (effective_radius - dist) / effective_radius + bottom,
                   img)
    return img

def get_max_radiance_field(imgs):
    """获取多个辐射场的最大值"""
    img = np.max(imgs, axis=0)
    return img

def build_ridge(A):
    """将路径转换为ridge image"""
    # 将 A 的第一个点对齐到 (10,10)
    A = A - A[0] + np.array([10, 10])

    # 对每个点计算辐射场图像
    imgs = []
    for i in range(A.shape[0]):
        imgs.append(build_radiance_field(A[i]))
    imgs = np.array(imgs)
    
    img = get_max_radiance_field(imgs)
    return img

def build_ridge_vmap(paths):
    """批量处理多条路径"""
    ridge_images = []
    for i in range(paths.shape[0]):
        ridge_images.append(build_ridge(paths[i]))
    return np.array(ridge_images)


# ==================== CCA 分析函数 ====================

def canoncorr(X0: np.array, Y0: np.array, fullReturn: bool = False) -> np.array:
    """
    Canonical Correlation Analysis (CCA) with added diagnostics and preprocessing
    
    Parameters:
    X, Y: (samples/observations) x (features) matrix
    fullReturn: whether all outputs should be returned or just `r` be returned
    
    Returns: A, B, r, U, V (if fullReturn is True) or just r (if fullReturn is False)
    A, B: Canonical coefficients for X and Y
    U, V: Canonical scores for the variables X and Y
    r: Canonical correlations
    """
    n, p1 = X0.shape
    p2 = Y0.shape[1]
    
    # Data diagnostics
    print(f"X shape: {X0.shape}, Y shape: {Y0.shape}")
    print(f"X condition number: {np.linalg.cond(X0)}")
    print(f"Y condition number: {np.linalg.cond(Y0)}")
    
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Preprocessing: Standardize the variables
    X = (X0 - np.mean(X0, 0)) / np.std(X0, 0)
    Y = (Y0 - np.mean(Y0, 0)) / np.std(Y0, 0)

    print(np.std(X0, 0), np.std(Y0, 0))

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1, T11, perm1 = qr(X, mode='economic', pivoting=True)
    Q2, T22, perm2 = qr(Y, mode='economic', pivoting=True)

    # Determine ranks with a more stringent threshold
    tol = np.finfo(float).eps * 100  # Increased tolerance
    rankX = np.sum(np.abs(np.diagonal(T11)) > tol * np.abs(T11[0, 0]))
    rankY = np.sum(np.abs(np.diagonal(T22)) > tol * np.abs(T22[0, 0]))

    print(f"Rank of X: {rankX}, Rank of Y: {rankY}")

    if rankX == 0:
        raise ValueError('X has zero rank')
    elif rankX < p1:
        logging.warning('X is not full rank')
        Q1 = Q1[:, :rankX]
        T11 = T11[:rankX, :rankX]

    if rankY == 0:
        raise ValueError('Y has zero rank')
    elif rankY < p2:
        logging.warning('Y is not full rank')
        Q2 = Q2[:, :rankY]
        T22 = T22[:rankY, :rankY]

    # Compute canonical coefficients and canonical correlations
    d = min(rankX, rankY)
    L,D,M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    A = inv(T11) @ L[:, :d] * np.sqrt(n - 1)
    B = inv(T22) @ M[:, :d] * np.sqrt(n - 1)
    r = D[:d]
    
    # Remove roundoff errors
    r = np.clip(r, 0, 1)

    if not fullReturn:
        return r

    # Put coefficients back to their full size and correct order
    A_full = np.zeros((p1, d))
    A_full[perm1, :] = np.vstack((A, np.zeros((p1 - rankX, d))))
    
    B_full = np.zeros((p2, d))
    B_full[perm2, :] = np.vstack((B, np.zeros((p2 - rankY, d))))

    # Compute the canonical variates
    U = X @ A_full
    V = Y @ B_full

    return A_full, B_full, r, U, V


# ==================== 可视化函数 ====================

def plot_lollipop(scores, title="Lollipop Plot of Scores", xlabel="Index", ylabel="Score", figsize=(12, 6)):
    """
    绘制带连线和数值标注的棒棒糖图，数值保留两位小数，x轴标签从1开始
    
    参数:
    scores : list 或 numpy array
        要绘制的数据
    title : str, 可选
        图表标题
    xlabel : str, 可选
        x轴标签
    ylabel : str, 可选
        y轴标签
    figsize : tuple, 可选
        图表大小
    """
    
    # 创建对应的 x 值
    x = np.arange(len(scores))

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制垂直线
    ax.vlines(x, 0, scores, colors='gray', lw=1, alpha=0.5)

    # 绘制数据点并连线
    ax.plot(x, scores, color='black', marker='o', markersize=8, linestyle='-', linewidth=1)

    # 添加数值标注，保留两位小数
    for i, score in enumerate(scores):
        ax.annotate(f'{score:.2f}', (i, score), textcoords="offset points", 
                    xytext=(0,10), ha='center', va='bottom', fontsize=8)

    # 设置坐标轴范围
    ax.set_xlim(-0.5, len(scores) - 0.5)
    ax.set_ylim(0, max(scores) * 1.2)  # 增加上界以容纳标注

    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 移除网格线
    ax.grid(False)

    # 调整 x 轴刻度，标签从1开始
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, len(scores) + 1))

    # 设置 y 范围 0-1
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('lollipop_plot.png', dpi=300, bbox_inches='tight')
    print("Lollipop plot saved as 'lollipop_plot.png'")
    plt.show()


# ==================== 主测试程序 ====================

def main():
    print("=" * 60)
    print("测试程序：Ridge Image 转换和 CCA 分析")
    print("=" * 60)
    
    # 1. 生成随机2D整数坐标路径（10个点，坐标范围0-10）
    print("\n[步骤1] 生成随机坐标路径")
    np.random.seed(42)  # 设置随机种子以便复现
    path = np.random.randint(0, 11, size=(10, 2))
    print(f"随机生成的坐标路径:\n{path}")
    print(f"路径形状: {path.shape}")
    
    # 2. 将路径转换为ridge image
    print("\n[步骤2] 将路径转换为 ridge image")
    ridge_img = build_ridge(path)
    print(f"Ridge image 形状: {ridge_img.shape}")
    print(f"Ridge image 最大值: {np.max(ridge_img):.4f}")
    print(f"Ridge image 最小值: {np.min(ridge_img):.4f}")
    
    # 可视化ridge image
    plt.figure(figsize=(6, 6))
    plt.imshow(ridge_img, cmap='hot', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Ridge Image from Random Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    # 在图上标注路径点
    aligned_path = path - path[0] + np.array([10, 10])
    plt.plot(aligned_path[:, 1], aligned_path[:, 0], 'b.-', linewidth=2, markersize=8, label='Path')
    plt.legend()
    plt.savefig('ridge_image.png', dpi=300, bbox_inches='tight')
    print("Ridge image 保存为 'ridge_image.png'")
    plt.show()
    
    # 3. 构造两组不同维度的随机数据
    print("\n[步骤3] 构造两组不同维度的数据")
    n_samples = 5000  # 样本数量（增加以避免过拟合）
    dim1 = 128  # 第一组数据维度（减少维度）
    dim2 = 441  # 第二组数据维度（减少维度）
    
    # 生成随机数据
    np.random.seed(123)
    X = np.random.randn(n_samples, dim1)
    Y = np.random.randn(n_samples, dim2)
    
    print(f"数据集 X 形状: {X.shape}")
    print(f"数据集 Y 形状: {Y.shape}")
    
    # 4. 执行CCA分析
    print("\n[步骤4] 执行 CCA 分析")
    A, B, r, U, V = canoncorr(X, Y, fullReturn=True)
    
    print(f"\n典型相关系数前10个: {r[:10]}")
    print(f"A (X的典型系数) 形状: {A.shape}")
    print(f"B (Y的典型系数) 形状: {B.shape}")
    print(f"U (X的典型得分) 形状: {U.shape}")
    print(f"V (Y的典型得分) 形状: {V.shape}")
    
    # 5. 绘制棒棒糖图
    print("\n[步骤5] 绘制棒棒糖图")
    scores = r[:min(10, len(r))]  # 取前10个典型相关系数
    plot_lollipop(scores, 
                  title="Canonical Correlation Analysis (CCA) Test", 
                  xlabel="Canonical Mode", 
                  ylabel="Correlation")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("生成的文件：")
    print("  - ridge_image.png (路径的ridge图像)")
    print("  - lollipop_plot.png (CCA相关系数图)")
    print("=" * 60)


if __name__ == "__main__":
    main()