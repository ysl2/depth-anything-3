import open3d as o3d
import numpy as np

# 读取iPhone 11预测结果点云
pcd = o3d.io.read_point_cloud("./output/iphone11-exp/exp_c/pcd/combined_pcd.ply")
print(f"Loaded {len(pcd.points)} points")

# 获取点云边界框进行居中显示
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
pcd.translate(-center)

# 修正上下颠倒：绕X轴旋转180度
R = pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
pcd.rotate(R, center=[0, 0, 0])

# 可视化
o3d.visualization.draw_geometries(
    [pcd],
    window_name="iPhone 11 - 3D Reconstruction (IMG_7375.MOV)",
    width=1920,
    height=1080
)
