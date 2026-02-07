#!/usr/bin/env python3
"""
语义点云可视化脚本

使用trimesh和matplotlib创建点云的可视化
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 尝试导入trimesh
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("trimesh not installed, install with: pip install trimesh")


def visualize_point_cloud(ply_path, sample_size=50000, mode='rgb'):
    """
    可视化语义点云

    Args:
        ply_path: PLY文件路径
        sample_size: 采样点数量（避免点太多）
        mode: 'rgb' 或 'semantic'
    """
    if HAS_TRIMESH:
        # 使用trimesh加载
        pc = trimesh.load(ply_path)
        points = pc.vertices

        # 获取颜色
        if hasattr(pc, 'visual'):
            colors = pc.visual.vertex_colors
        elif hasattr(pc, 'colors'):
            colors = pc.colors
        else:
            colors = None

        # 获取语义标签和语义颜色
        if hasattr(pc, 'vertex_attributes') and 'semantic_label' in pc.vertex_attributes:
            semantics = pc.vertex_attributes['semantic_label']
        elif hasattr(pc, 'semantic_label'):
            semantics = pc.semantic_label
        else:
            semantics = None

        if hasattr(pc, 'vertex_attributes') and 'semantic_red' in pc.vertex_attributes:
            sem_colors = np.stack([
                pc.vertex_attributes['semantic_red'],
                pc.vertex_attributes['semantic_green'],
                pc.vertex_attributes['semantic_blue'],
            ], axis=1)
        elif hasattr(pc, 'semantic_red'):
            sem_colors = np.stack([
                pc.semantic_red,
                pc.semantic_green,
                pc.semantic_blue,
            ], axis=1)
        else:
            sem_colors = None
    else:
        print("Cannot load PLY file without trimesh")
        return

    print(f"Point cloud loaded: {points.shape[0]} points")

    # 采样
    if len(points) > sample_size:
        indices = np.random.choice(len(points), sample_size, replace=False)
        points = points[indices]
        if colors is not None:
            colors = colors[indices]
        if semantics is not None:
            semantics = semantics[indices]
        if sem_colors is not None:
            sem_colors = sem_colors[indices]

    # 选择颜色
    if mode == 'semantic' and sem_colors is not None:
        display_colors = sem_colors / 255.0
        title_suffix = " (Semantic Colors)"
    elif colors is not None:
        display_colors = colors / 255.0
        title_suffix = " (RGB Colors)"
    else:
        display_colors = np.ones_like(points) * 0.5
        title_suffix = " (No Color)"

    # 创建3D图
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=display_colors,
        s=0.1,
        alpha=0.6
    )

    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Semantic Point Cloud{title_suffix}\n{len(points)} points (sampled from {pc.vertices.shape[0]} total)')

    # 设置视角
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    return fig


def visualize_semantic_masks(preview_path):
    """显示语义掩码预览图"""
    img = plt.imread(preview_path)
    plt.figure(figsize=(16, 8))
    plt.imshow(img)
    plt.title("Semantic Segmentation Preview")
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()


def main():
    output_dir = "./semantic_test_output"

    print("=" * 70)
    print("Semantic Point Cloud Visualization")
    print("=" * 70)
    print()

    # 显示语义掩码预览
    semantic_preview_path = os.path.join(output_dir, "semantic_preview.png")
    if os.path.exists(semantic_preview_path):
        print("Opening semantic segmentation preview...")
        fig1 = visualize_semantic_masks(semantic_preview_path)
        plt.show()

    # 显示点云
    ply_path = os.path.join(output_dir, "semantic_pointcloud.ply")
    if os.path.exists(ply_path):
        print()
        print("Generating 3D point cloud visualization...")
        print("Note: This may take a moment for large point clouds")

        # RGB模式
        print("\n1. RGB Color Mode")
        fig2 = visualize_point_cloud(ply_path, mode='rgb')
        plt.show()

        # 语义颜色模式
        print("\n2. Semantic Color Mode")
        fig3 = visualize_point_cloud(ply_path, mode='semantic')
        plt.show()

        print("\nVisualization complete!")
        print()
        print("For better visualization, use:")
        print("  - CloudCompare (https://www.cloudcompare.org/)")
        print("  - MeshLab (http://www.meshlab.net/)")


if __name__ == "__main__":
    main()
