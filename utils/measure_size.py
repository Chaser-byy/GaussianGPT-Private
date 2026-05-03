# import open3d as o3d
# import os

# def calculate_ply_dimensions(file_path, to_meters_scale=1.0):
#     """
#     读取 PLY 文件并以米为单位输出其尺寸。
    
#     参数:
#     file_path (str): PLY 文件的相对或绝对路径。
#     to_meters_scale (float): 转换到米的比例系数。
#     """
#     if not os.path.exists(file_path):
#         print(f"错误: 找不到文件 '{file_path}'")
#         return

#     print(f"正在加载模型: {file_path} ...")
    
#     # 修正：使用 read_point_cloud 明确读取点云数据
#     pcd = o3d.io.read_point_cloud(file_path)
    
#     if pcd.is_empty():
#         print("错误: 文件读取失败或文件为空。请确保它是一个有效的 PLY 文件。")
#         return

#     # 获取轴对齐包围盒 (Axis-Aligned Bounding Box)
#     aabb = pcd.get_axis_aligned_bounding_box()
    
#     # get_extent() 返回一个包含 [X, Y, Z] 方向上长度的数组
#     extent = aabb.get_extent()
    
#     # 根据比例系数转换为米
#     size_x = extent[0] * to_meters_scale
#     size_y = extent[1] * to_meters_scale
#     size_z = extent[2] * to_meters_scale
    
#     print("-" * 30)
#     print("包围盒尺寸 (以米为单位):")
#     print(f"X 轴宽度: {size_x:.6f} m")
#     print(f"Y 轴高度: {size_y:.6f} m")
#     print(f"Z 轴深度: {size_z:.6f} m")
#     print("-" * 30)

# if __name__ == "__main__":
#     # 替换为你实际的 PLY 文件路径
#     ply_file = "/home/chengtianle/GaussianGPT/data/overfit/001_clean.ply" 
    
#     # 比例系数：如果高斯抛溅训练时的场景本身就是基于米的单位，保持 1.0 即可
#     scale_factor = 1.0  
    
#     calculate_ply_dimensions(ply_file, to_meters_scale=scale_factor)

import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_ply_distribution(file_path, save_path, to_meters_scale=1.0):
    """
    读取 PLY 文件，提取坐标，并绘制保存 X, Y, Z 坐标的分布图。
    
    参数:
    file_path (str): PLY 文件的路径。
    save_path (str): 保存分布图图片的路径 (例如: 'distribution.png')。
    to_meters_scale (float): 转换到米的比例系数。
    """
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 '{file_path}'")
        return

    print(f"正在加载模型: {file_path} ...")
    pcd = o3d.io.read_point_cloud(file_path)
    
    if pcd.is_empty():
        print("错误: 文件读取失败或文件为空。请确保它是一个有效的 PLY 文件。")
        return

    # 1. 提取所有点的坐标并转换为 numpy 数组
    # pcd.points 包含了所有的 (X, Y, Z) 坐标
    points = np.asarray(pcd.points) * to_meters_scale
    
    # 分离 X, Y, Z 坐标
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]

    print(f"成功提取 {len(points)} 个点。正在绘制分布图...")

    # 2. 使用 Matplotlib 绘制分布图
    # 创建一个 1行3列 的画布，尺寸为 18x5 英寸
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 设置直方图的柱子数量 (bins)
    bins_count = 100

    # 绘制 X 轴分布 (红色)
    axes[0].hist(x_coords, bins=bins_count, color='indianred', alpha=0.8)
    axes[0].set_title('X Coordinate Distribution', fontsize=14)
    axes[0].set_xlabel('X Position (meters)', fontsize=12)
    axes[0].set_ylabel('Number of Points', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 绘制 Y 轴分布 (绿色)
    axes[1].hist(y_coords, bins=bins_count, color='mediumseagreen', alpha=0.8)
    axes[1].set_title('Y Coordinate Distribution', fontsize=14)
    axes[1].set_xlabel('Y Position (meters)', fontsize=12)
    axes[1].set_ylabel('Number of Points', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # 绘制 Z 轴分布 (蓝色)
    axes[2].hist(z_coords, bins=bins_count, color='cornflowerblue', alpha=0.8)
    axes[2].set_title('Z Coordinate Distribution', fontsize=14)
    axes[2].set_xlabel('Z Position (meters)', fontsize=12)
    axes[2].set_ylabel('Number of Points', fontsize=12)
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    # 调整布局以防标签重叠
    plt.tight_layout()

    # 3. 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"分布图已成功保存至: {save_path}")
    
    # 清理内存
    plt.close()

if __name__ == "__main__":
    # 输入的 PLY 文件路径
    ply_file = "/home/chengtianle/GaussianGPT/data/overfit/008_clean.ply" 
    
    # 想要保存的图片路径
    output_image = "/home/chengtianle/GaussianGPT/data/overfit/008_clean_distribution.png"
    
    scale_factor = 1.0  
    
    plot_and_save_ply_distribution(ply_file, output_image, to_meters_scale=scale_factor)