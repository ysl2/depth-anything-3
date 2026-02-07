#!/usr/bin/env python3
"""
打开语义点云可视化结果
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud():
    """可视化语义点云"""
    ply_path = "./semantic_test_output/semantic_pointcloud.ply"

    # 检查trimesh是否可用
    try:
        import trimesh
        pc = trimesh.load(ply_path)
        points = pc.vertices

        # 获取颜色
        if hasattr(pc, 'visual') and pc.visual is not None:
            colors = pc.visual.vertex_colors
        elif hasattr(pc, 'colors'):
            colors = pc.colors
        else:
            colors = None

        print(f"点云加载成功！")
        print(f"点数: {len(points)}")

    except ImportError:
        print("trimesh未安装，尝试手动解析PLY...")
        # 手动解析PLY文件
        with open(ply_path, 'rb') as f:
            lines = []
            for _ in range(20):
                line = f.readline().decode('ascii').strip()
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[-1])
                    print(f"顶点数: {num_vertices}")
                elif line.startswith('end_header'):
                    break

            # 读取二进制数据
            import struct
            dtype = np.dtype([
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('red', np.uint8),
                ('green', np.uint8),
                ('blue', np.uint8),
                ('label', np.uint8),
                ('sem_red', np.uint8),
                ('sem_green', np.uint8),
                ('sem_blue', np.uint8),
            ])

            data = np.fromfile(f, dtype=dtype)
            points = np.stack([data['x'], data['y'], data['z']], axis=1)
            colors = np.stack([data['red'], data['green'], data['blue']], axis=1)
            sem_colors = np.stack([data['sem_red'], data['sem_green'], data['sem_blue']], axis=1)
            print(f"点云加载成功！点数: {len(points)}")

    # 采样点（避免点太多）
    sample_size = min(50000, len(points))
    indices = np.random.choice(len(points), sample_size, replace=False)
    points = points[indices]
    if colors is not None:
        colors = colors[indices] / 255.0

    # 创建3D图
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制点云
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors if colors is not None else 'gray',
        s=0.5,
        alpha=0.5
    )

    # 设置标签
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title(f'语义点云可视化 ({sample_size} 采样点 / {len(points)} 总点)', fontsize=14)

    # 设置视角
    ax.view_init(elev=30, azim=45)

    # 设置背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

    plt.tight_layout()

    print("正在打开可视化窗口...")
    plt.show()

    return fig

def show_previews():
    """显示预览图"""
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 原图+深度预览
    preview_path = "./semantic_test_output/preview.png"
    if os.path.exists(preview_path):
        img1 = mpimg.imread(preview_path)
        axes[0].imshow(img1)
        axes[0].set_title('原图 + 深度图', fontsize=14)
        axes[0].axis('off')

    # 语义掩码预览
    semantic_path = "./semantic_test_output/semantic_preview.png"
    if os.path.exists(semantic_path):
        img2 = mpimg.imread(semantic_path)
        axes[1].imshow(img2)
        axes[1].set_title('语义分割结果', fontsize=14)
        axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="打开语义点云可视化")
    parser.add_argument("--mode", choices=["3d", "preview", "both"], default="both",
                       help="可视化模式: 3d点云、预览图、或两者都显示")

    args = parser.parse_args()

    if args.mode in ["preview", "both"]:
        print("显示预览图...")
        show_previews()

    if args.mode in ["3d", "both"]:
        print("打开3D点云可视化...")
        visualize_point_cloud()
