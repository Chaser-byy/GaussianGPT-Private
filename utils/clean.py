# import open3d as o3d
# import os

# def clean_ply_with_open3d(file_path, save_path):
#     print(f"正在加载模型: {file_path} ...")
#     pcd = o3d.io.read_point_cloud(file_path)
    
#     print(f"清理前点数: {len(pcd.points)}")

#     # ==========================================
#     # 核心：Statistical Outlier Removal (SOR)
#     # nb_neighbors: 算每个点时，考虑它周围的多少个邻居（一般设为 20-50）
#     # std_ratio: 阈值。值越小，删得越狠。2.0 表示距离大于平均距离 2 个标准差的视为离群点
#     # ==========================================
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
#     # 根据保留下来的索引 (ind) 提取干净的点云
#     clean_pcd = pcd.select_by_index(ind)
    
#     print(f"清理后点数: {len(clean_pcd.points)}")
#     print(f"共剔除离群点: {len(pcd.points) - len(clean_pcd.points)} 个")

#     # 保存清理后的模型
#     o3d.io.write_point_cloud(save_path, clean_pcd)
#     print(f"清理后的模型已保存至: {save_path}")

# # 运行示例
# ply_file = "/home/chengtianle/GaussianGPT/data/overfit/train/004.ply" 
# clean_file = "/home/chengtianle/GaussianGPT/data/overfit/004_clean.ply" 
# clean_ply_with_open3d(ply_file, clean_file)

# import numpy as np
# from plyfile import PlyData, PlyElement
# import os

# def clean_3dgs_by_percentile(file_path, save_path, lower_pct=1.0, upper_pct=99.0):
#     """
#     基于百分位数清理 3DGS 的 PLY 文件，并完美保留所有 Gaussian 属性。
#     """
#     if not os.path.exists(file_path):
#         print(f"错误: 找不到文件 '{file_path}'")
#         return

#     print(f"正在读取 3DGS 模型: {file_path} ... (这可能需要一点时间)")
    
#     # 1. 使用 plyfile 读取原始数据
#     plydata = PlyData.read(file_path)
    
#     # 提取名为 'vertex' 的元素数据 (3DGS 的所有数据都在这一个结构化数组里)
#     vertex_data = plydata.elements[0].data
    
#     print(f"清理前高斯点数: {len(vertex_data)}")

#     # 2. 提取 X, Y, Z 坐标用于计算离群值
#     x = vertex_data['x']
#     y = vertex_data['y']
#     z = vertex_data['z']
    
#     # 将其组合成形状为 (N, 3) 的数组，方便复用你之前的逻辑
#     points = np.vstack((x, y, z)).T

#     # 3. 计算 1% 和 99% 的分界线
#     lower_bounds = np.percentile(points, lower_pct, axis=0)
#     upper_bounds = np.percentile(points, upper_pct, axis=0)

#     # 4. 生成掩码 (Mask)
#     mask = (points >= lower_bounds) & (points <= upper_bounds)
#     valid_mask = mask.all(axis=1) # 必须 3 个轴都合法
    
#     # 5. 【核心步骤】直接使用布尔掩码过滤结构化数组
#     # 这样不仅过滤了位置，还连带把这行的 color, scale, rot, sh 等属性一起保留了
#     clean_vertex_data = vertex_data[valid_mask]
    
#     print(f"清理后高斯点数: {len(clean_vertex_data)}")
#     print(f"共剔除离群点: {len(vertex_data) - len(clean_vertex_data)} 个")

#     # 6. 重新打包并保存为 PLY
#     print("正在保存清理后的 3DGS 模型...")
#     # 构建新的 PlyElement
#     clean_element = PlyElement.describe(clean_vertex_data, 'vertex')
    
#     # 继承原文件的文本/二进制格式设置，生成新的 PlyData
#     clean_plydata = PlyData([clean_element], text=plydata.text, byte_order=plydata.byte_order)
    
#     # 写入文件
#     clean_plydata.write(save_path)
#     print(f"成功保存至: {save_path}")

# if __name__ == "__main__":
#     ply_file = "/home/chengtianle/GaussianGPT/data/overfit/train/002.ply"
#     clean_file = "/home/chengtianle/GaussianGPT/data/overfit/002_clean.ply" 
    
#     # 运行清理
#     clean_3dgs_by_percentile(ply_file, clean_file, lower_pct=20.0, upper_pct=85.0)

import numpy as np
from plyfile import PlyData, PlyElement
import os

def clean_3dgs_by_xyz_ranges(file_path, save_path, 
                             x_min=-np.inf, x_max=np.inf,
                             y_min=-np.inf, y_max=np.inf,
                             z_min=-np.inf, z_max=np.inf):
    """
    通过指定 X、Y、Z 轴的绝对数值范围（包围盒）来裁剪 3DGS 模型。
    没有指定的边界默认保留到无穷远。
    """
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 '{file_path}'")
        return

    print(f"正在读取 3DGS 模型: {file_path} ...")
    
    # 1. 读取原始数据
    plydata = PlyData.read(file_path)
    vertex_data = plydata.elements[0].data
    
    print(f"清理前高斯点数: {len(vertex_data)}")

    # 2. 提取 X, Y, Z 坐标
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']

    # 3. 【核心步骤】分别计算三个轴的掩码，然后求交集 (&)
    mask_x = (x >= x_min) & (x <= x_max)
    mask_y = (y >= y_min) & (y <= y_max)
    mask_z = (z >= z_min) & (z <= z_max)
    
    # 只有当一个点在 X, Y, Z 三个方向上都符合要求时，才保留它
    valid_mask = mask_x & mask_y & mask_z
    
    # 4. 过滤数据
    clean_vertex_data = vertex_data[valid_mask]
    
    print(f"清理后高斯点数: {len(clean_vertex_data)}")
    print(f"共剔除点数: {len(vertex_data) - len(clean_vertex_data)} 个")

    if len(clean_vertex_data) == 0:
        print("警告: 筛选后没有剩余的点！请检查你设置的 XYZ 范围是否太小或没有交集。")
        return

    # 5. 保存
    print("正在保存裁剪后的 3DGS 模型...")
    clean_element = PlyElement.describe(clean_vertex_data, 'vertex')
    clean_plydata = PlyData([clean_element], text=plydata.text, byte_order=plydata.byte_order)
    clean_plydata.write(save_path)
    print(f"成功保存至: {save_path}")


if __name__ == "__main__":
    ply_file = "/home/chengtianle/GaussianGPT/data/overfit/train/008.ply"
    clean_file = "/home/chengtianle/GaussianGPT/data/overfit/008_clean.ply" 
    
    # ==========================================
    # 在这里填入你想保留的 X, Y, Z 范围。
    # 建议对照你之前生成的 3 张分布图来填写。
    # 如果某个方向你不想切（保留原样），就写 -np.inf 或 np.inf
    # ==========================================
    
    # 示例：假设我们想切掉 X 轴两侧极端的点，Y 轴切掉上面，Z 轴切掉地面以下
    TARGET_X_MIN = -3.6
    TARGET_X_MAX = 12.2
    
    TARGET_Y_MIN = -19.5
    TARGET_Y_MAX = 2  # Y轴上限不限制
    
    TARGET_Z_MIN = -0.5
    TARGET_Z_MAX = 3.5
    
    # 运行裁剪
    clean_3dgs_by_xyz_ranges(
        ply_file, clean_file,
        x_min=TARGET_X_MIN, x_max=TARGET_X_MAX,
        y_min=TARGET_Y_MIN, y_max=TARGET_Y_MAX,
        z_min=TARGET_Z_MIN, z_max=TARGET_Z_MAX
    )