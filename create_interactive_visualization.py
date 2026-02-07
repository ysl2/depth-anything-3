#!/usr/bin/env python3
"""
创建交互式3D语义点云可视化
使用Plotly创建可以在浏览器中直接查看的3D可视化
"""
import os
import sys
import numpy as np

# 添加src目录到路径
sys.path.insert(0, 'src')

def create_interactive_3d_visualization():
    """创建交互式3D语义点云可视化"""
    import plotly.graph_objects as go

    ply_path = "./semantic_test_output/semantic_pointcloud.ply"
    output_path = "./semantic_test_output/interactive_3d.html"

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
        print("trimesh未安装，安装中...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'trimesh', '-q'])
        import trimesh
        pc = trimesh.load(ply_path)
        points = pc.vertices
        colors = pc.colors if hasattr(pc, 'colors') else None
        print(f"点云加载成功！点数: {len(points)}")

    # 采样点（Plotly可以处理更多点）
    sample_size = min(200000, len(points))
    indices = np.random.choice(len(points), sample_size, replace=False)
    points = points[indices]

    if colors is not None:
        colors = colors[indices]

    print(f"创建可视化，使用 {sample_size} 个采样点...")

    # 创建3D散点图
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors / 255.0 if colors is not None else 'lightblue',
            opacity=0.6,
            line=dict(width=0)
        ),
        text=[f'Point {i}' for i in range(len(points))]
    )])

    fig.update_layout(
        title='三维语义点云交互式可视化',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # 保存为HTML
    fig.write_html(output_path)
    print(f"交互式3D可视化已保存到: {output_path}")

    return output_path


def main():
    output_path = create_interactive_3d_visualization()

    print()
    print("=" * 60)
    print("✅ 交互式3D可视化已创建！")
    print("=" * 60)
    print()
    print("请打开以下链接查看3D语义点云：")
    print()
    print(f"  file://{os.path.abspath(output_path)}")
    print()
    print("或者在浏览器中打开：")
    print(f"  http://localhost:8000/interactive_3d.html")
    print()
    print("操作说明：")
    print("  - 鼠标拖拽：旋转视角")
    print("  - 滚轮：缩放")
    print("  - 右键拖拽：平移")
    print("  - 点击点：查看点信息")


if __name__ == "__main__":
    main()
