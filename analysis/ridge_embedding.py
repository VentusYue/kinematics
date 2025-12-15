
import numpy as np

def build_radiance_field(p, grid_size=21, center=10, scale=1.414, top=10.0, bottom=0.0):
    """构建单个点的辐射场"""
    img = np.zeros((grid_size, grid_size))
    effective_radius = grid_size * scale
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
    if len(imgs) == 0:
        return None
    img = np.max(imgs, axis=0)
    return img

def build_ridge(path, grid_size=21):
    """
    将路径转换为ridge image
    path: (T, 2) array of coordinates
    """
    if len(path) == 0:
        return np.zeros((grid_size, grid_size))
        
    # Copy path to avoid modifying input
    A = path.copy()
    
    # 将 A 的第一个点对齐到中心 (10,10) for 21x21
    center = grid_size // 2
    offset = np.array([center, center]) - A[0]
    A = A + offset

    # 对每个点计算辐射场图像
    imgs = []
    for i in range(A.shape[0]):
        imgs.append(build_radiance_field(A[i], grid_size=grid_size, center=center))
    imgs = np.array(imgs)
    
    img = get_max_radiance_field(imgs)
    return img

def build_ridge_vector(path, grid_size=21):
    """
    Return flattened ridge image
    """
    img = build_ridge(path, grid_size)
    return img.flatten()

