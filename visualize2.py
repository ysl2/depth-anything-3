"""
Feature 3DGS Point Cloud Visualizer
支持 3D Gaussian Splatting 格式的 PLY 文件可视化
"""
from plyfile import PlyData
import numpy as np
import open3d as o3d

# 读取 ply 文件
print("Loading 3DGS point cloud...")
plydata = PlyData.read("./point_cloud.ply")
vertex = plydata['vertex']

# 提取数据
xyz = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
f_dc = np.vstack((vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2'])).T
opacity = vertex['opacity']

print(f"Loaded {len(xyz)} points")

# 从球谐 DC 系数计算颜色: RGB = SH0 * 0.5 + 0.5
colors = f_dc * 0.5 + 0.5
colors = np.clip(colors, 0, 1)

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

# 获取点云边界框进行居中显示
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
pcd.translate(-center)

# 修正上下颠倒：绕X轴旋转180度
R = pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
pcd.rotate(R, center=[0, 0, 0])

# 可视化
print("Opening interactive viewer...")
print("Controls:")
print("  - Mouse drag: Rotate view")
print("  - Mouse wheel: Zoom")
print("  - Shift + drag: Pan")

vis = o3d.visualization.Visualizer()
vis.create_window(
    window_name="Feature 3DGS Point Cloud",
    width=1920,
    height=1080
)

# 设置渲染选项
opt = vis.get_render_option()
opt.background_color = np.asarray([1.0, 1.0, 1.0])  # 白色背景
opt.point_size = 2.0

vis.add_geometry(pcd)
vis.run()
vis.destroy_window()
