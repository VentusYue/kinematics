
import numpy as np

def build_radiance_field(
    p,
    grid_size: int = 21,
    top: float = 10.0,
    bottom: float = 0.0,
    radius_scale: float = 1.414,
) -> np.ndarray:
    """
    Build a single-point radiance field on a 2D grid.

    Key detail (per reference code): use a *global* effective radius so the cone
    covers the whole 21x21 grid, reducing constant/zero pixels.
    """
    img = np.zeros((grid_size, grid_size), dtype=np.float32)
    effective_radius = float(grid_size) * float(radius_scale)  # ~29.7 for 21*1.414
    px, py = float(p[0]), float(p[1])

    xs = np.arange(grid_size, dtype=np.float32)[:, None]
    ys = np.arange(grid_size, dtype=np.float32)[None, :]
    dist = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)

    img = np.where(
        dist < effective_radius,
        (top - bottom) * (effective_radius - dist) / effective_radius + bottom,
        0.0,
    ).astype(np.float32)
    return img

def get_max_radiance_field(imgs):
    """获取多个辐射场的最大值"""
    if len(imgs) == 0:
        return None
    img = np.max(imgs, axis=0)
    return img


def get_sum_radiance_field(imgs, clip_top=10.0):
    """获取多个辐射场的求和（可选 clip）"""
    if len(imgs) == 0:
        return None
    img = np.sum(imgs, axis=0)
    if clip_top is not None:
        img = np.clip(img, 0.0, clip_top)
    return img


def normalize_path_to_grid(path: np.ndarray, grid_size: int = 21, margin: int = 1) -> np.ndarray:
    """
    将路径（以起点为原点的 tile 坐标）缩放到能落在 ridge grid 的范围内，减少全零/常数像素列。
    - 输入 path: (T,2)
    - 输出仍然是以起点为原点的坐标；后续 build_ridge 会再把起点平移到 grid center。
    """
    if path is None or len(path) == 0:
        return path

    A = np.asarray(path, dtype=np.float32).copy()
    # ensure first point is origin
    A = A - A[0]

    max_abs = float(np.max(np.abs(A)))
    if not np.isfinite(max_abs) or max_abs < 1e-6:
        return A

    center = grid_size // 2
    target = max(center - 1 - int(margin), 1)
    scale = target / max_abs
    return A * scale

def build_ridge(path, grid_size: int = 21, radius_scale: float = 1.414, aggregate: str = "max") -> np.ndarray:
    """
    将路径转换为ridge image
    path: (T, 2) array of coordinates
    """
    if len(path) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
        
    # Copy path to avoid modifying input
    A = np.asarray(path, dtype=np.float32).copy()
    
    # 将 A 的第一个点对齐到中心 (10,10) for 21x21
    center = grid_size // 2
    offset = np.array([center, center], dtype=np.float32) - A[0]
    A = A + offset

    # 对每个点计算辐射场图像
    imgs = []
    for i in range(A.shape[0]):
        imgs.append(build_radiance_field(A[i], grid_size=grid_size, radius_scale=radius_scale))
    imgs = np.stack(imgs, axis=0).astype(np.float32)
    
    if aggregate == "max":
        img = get_max_radiance_field(imgs)
    elif aggregate == "sum":
        # clip_top uses the same default top as build_radiance_field
        img = get_sum_radiance_field(imgs, clip_top=10.0)
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")
    return img

def build_ridge_vector(
    path,
    grid_size: int = 21,
    radius_scale: float = 1.414,
    aggregate: str = "max",
    normalize_path: bool = False,
) -> np.ndarray:
    """
    Return flattened ridge image
    """
    if normalize_path:
        path = normalize_path_to_grid(path, grid_size=grid_size)
    img = build_ridge(path, grid_size=grid_size, radius_scale=radius_scale, aggregate=aggregate)
    return img.reshape(-1).astype(np.float32)

