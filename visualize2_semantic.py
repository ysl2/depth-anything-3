"""
Feature 3DGS Point Cloud Visualizer - 语义分割模式
支持 3D Gaussian Splatting 格式的 PLY 文件可视化
"""
from plyfile import PlyData
import numpy as np
import open3d as o3d

# 读取 ply 文件
print("Loading 3DGS point cloud (Semantic mode)...")
plydata = PlyData.read("./point_cloud.ply")
vertex = plydata['vertex']

# 提取数据
xyz = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
opacity = vertex['opacity']

# 提取语义特征 (125个通道)
semantic_features = []
for i in range(125):
    semantic_features.append(vertex[f'semantic_{i}'])
semantic = np.vstack(semantic_features).T  # (N, 125)

print(f"Loaded {len(xyz)} points")
print(f"Semantic features: {semantic.shape[1]} channels")

# 计算语义颜色 (使用第一个语义通道)
semantic_channel = semantic[:, 0]
sem_min, sem_max = semantic_channel.min(), semantic_channel.max()
semantic_norm = (semantic_channel - sem_min) / (sem_max - sem_min)

# 使用 Spectral 颜色映射
import matplotlib.cm as cm
colors = cm.Spectral(semantic_norm)
colors = colors[:, :3]

# 过滤掉 opacity 太低的点
opacity_threshold = 0.5
valid_mask = opacity > opacity_threshold
print(f"Valid points after opacity filter: {valid_mask.sum()}")

xyz_filtered = xyz[valid_mask]
colors_filtered = colors[valid_mask]

# 创建 Open3D 点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz_filtered)
pcd.colors = o3d.utility.Vector3dVector(colors_filtered)

# 居中和旋转
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
pcd.translate(-center)
R = pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
pcd.rotate(R, center=[0, 0, 0])

# 可视化
print("Opening viewer (Semantic Segmentation mode)...")
vis = o3d.visualization.Visualizer()
vis.create_window("3DGS - Semantic Segmentation", 1920, 1080)

opt = vis.get_render_option()
opt.background_color = np.asarray([1, 1, 1])  # 白色背景
opt.point_size = 4.0

vis.add_geometry(pcd)
vis.run()
vis.destroy_window()
