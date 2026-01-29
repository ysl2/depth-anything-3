"""
DCIM 数据集 DA3-Streaming 3D 重建可视化脚本
"""
import open3d as o3d
import numpy as np
import sys

def main():
    print("=" * 60)
    print("DCIM 3D Reconstruction Visualization")
    print("=" * 60)

    # 读取点云
    pcd_path = "./output/dcim/pcd/combined_pcd.ply"
    print(f"\nLoading point cloud from: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)

    print(f"Points: {len(pcd.points):,}")
    print(f"Colors: {'Yes' if len(pcd.colors) > 0 else 'No'}")

    # 获取点云边界框
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()

    print(f"\nBounding Box:")
    print(f"  Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"  Extent: [{extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f}]")

    # 居中点云
    pcd.translate(-center)
    print(f"\nCentered point cloud to origin")

    # 可视化
    print("\n" + "=" * 60)
    print("Visualization Controls:")
    print("  - Left Mouse: Rotate")
    print("  - Right Mouse: Pan")
    print("  - Scroll: Zoom")
    print("  - H: Help (more options)")
    print("=" * 60)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="DCIM 3D Reconstruction (Depth-Anything-3)",
        width=1920,
        height=1080,
        left=50,
        top=50,
        point_show_normal=False
    )

if __name__ == "__main__":
    main()
