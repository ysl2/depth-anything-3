#!/usr/bin/env python3
"""
DCIM 3D点云可视化脚本
支持交互式3D可视化和2D投影展示
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def visualize_3d_interactive(pcd_path):
    """交互式3D可视化"""
    print(f"Loading point cloud from: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"Loaded {len(pcd.points)} points")

    # 获取点云边界框进行居中显示
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    pcd.translate(-center)

    # 修正上下颠倒：绕X轴旋转180度
    R = pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
    pcd.rotate(R, center=[0, 0, 0])

    print("\n=== 交互式3D可视化控制 ===")
    print("- 鼠标左键拖拽: 旋转视角")
    print("- 鼠标滚轮: 缩放")
    print("- Shift + 鼠标拖拽: 平移")
    print("- 按 'H': 查看更多帮助")
    print("- 按 'Q' 或关闭窗口: 退出")
    print()

    # 可视化
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="DCIM 3D Reconstruction (DJI Mini3 Pro)",
        width=1920,
        height=1080,
        left=50,
        top=50,
        point_show_normal=False
    )

def visualize_2d_projections(pcd_path, output_path="dcim_projections.png"):
    """生成2D投影图（用于展示）"""
    print(f"Loading point cloud from: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    print(f"Loaded {len(points)} points")

    # 居中显示
    center = points.mean(axis=0)
    points = points - center

    # 修正上下颠倒
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    points = points @ R.T

    # 创建三视图投影
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('DCIM 3D Reconstruction - Point Cloud Projections\nDJI Mini3 Pro | 969 images | 2,536,530 points',
                 fontsize=14, fontweight='bold')

    # XY平面投影（俯视图）
    ax = axes[0]
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=0.1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Top View (XY Projection)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # XZ平面投影（正视图）
    ax = axes[1]
    ax.scatter(points[:, 0], points[:, 2], c=colors, s=0.1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Front View (XZ Projection)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # YZ平面投影（侧视图）
    ax = axes[2]
    ax.scatter(points[:, 1], points[:, 2], c=colors, s=0.1, alpha=0.6)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('Side View (YZ Projection)')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"2D projections saved to: {output_path}")

    return fig


def main():
    pcd_path = "./output/dcim/pcd/combined_pcd.ply"

    print("=" * 60)
    print("DCIM 3D Point Cloud Visualization")
    print("=" * 60)
    print()

    # 生成2D投影图
    print("Generating 2D projections...")
    fig = visualize_2d_projections(pcd_path, "dcim_2d_projections.png")

    # 显示2D投影图
    print("\nDisplaying 2D projections...")
    plt.show()

    # 询问是否启动交互式3D可视化
    print("\n" + "=" * 60)
    response = input("Launch interactive 3D visualization? (y/n): ").lower()
    if response == 'y':
        visualize_3d_interactive(pcd_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
