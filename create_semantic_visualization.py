#!/usr/bin/env python3
"""
真正的语义分割3D可视化 - 显示语义标签而非RGB颜色
"""
import os
import sys
import numpy as np

def create_semantic_visualization():
    """创建显示语义标签的3D可视化"""

    # 检查plotly是否安装
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("安装plotly...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'plotly', '-q'])
        import plotly.graph_objects as go

    ply_path = "./semantic_test_output/semantic_pointcloud.ply"
    output_path = "./semantic_test_output/semantic_3d.html"

    # 读取PLY文件
    with open(ply_path, 'rb') as f:
        # 跳过头部
        for _ in range(20):
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break

        # 读取二进制数据
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('red', np.uint8),
            ('green', np.uint8),
            ('blue', np.uint8),
            ('semantic_label', np.uint8),
            ('semantic_red', np.uint8),
            ('semantic_green', np.uint8),
            ('semantic_blue', np.uint8),
        ])

        data = np.fromfile(f, dtype=dtype)

    print(f"点云加载成功！")
    print(f"总点数: {len(data)}")
    print(f"语义标签范围: {data['semantic_label'].min()} - {data['semantic_label'].max()}")

    # 提取坐标
    points = np.stack([data['x'], data['y'], data['z']], axis=1)

    # 使用语义RGB颜色（这是语义分割的结果！）
    semantic_colors = np.stack([
        data['semantic_red'],
        data['semantic_green'],
        data['semantic_blue']
    ], axis=1)

    # 检查语义标签分布
    unique_labels, counts = np.unique(data['semantic_label'], return_counts=True)
    print(f"\n语义类别分布:")
    for label, count in zip(unique_labels, counts):
        pct = count / len(data) * 100
        # 找出该类别的代表颜色
        mask = data['semantic_label'] == label
        if np.any(mask):
            sample_colors = semantic_colors[mask][0]
            print(f"  类别 {label}: {count:,} 点 ({pct:.2f}%) - RGB({sample_colors[0]}, {sample_colors[1]}, {sample_colors[2]})")

    # 采样点（20万点）
    sample_size = min(200000, len(points))
    indices = np.random.choice(len(points), sample_size, replace=False)
    points = points[indices]
    semantic_colors = semantic_colors[indices]
    semantic_labels = data['semantic_label'][indices]

    print(f"\n创建可视化，使用 {sample_size} 个采样点...")

    # 创建3D散点图
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=semantic_colors / 255.0,  # 使用语义颜色！
            opacity=0.6,
            line=dict(width=0)
        ),
        text=[f'Label: {label}' for label in semantic_labels]
    )])

    # 根据语义类别设置颜色映射标题
    label_names = {
        0: '背景/建筑',
        1: '检测物体'
    }
    label_text = ', '.join([f"{label_names.get(l, l)} ({c/len(data)*100:.1f}%)"
                        for l, c in zip(unique_labels, counts)])

    fig.update_layout(
        title=f'三维语义分割可视化 - {label_text}',
        scene=dict(
            xaxis=dict(title='X', backgroundcolor='rgb(20,20,20)'),
            yaxis=dict(title='Y', backgroundcolor='rgb(20,20,20)'),
            zaxis=dict(title='Z', backgroundcolor='rgb(20,20,20)'),
            bgcolor='rgb(10,10,10)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='rgb(10,10,10)',
        font=dict(color='white')
    )

    # 保存为HTML
    fig.write_html(output_path)
    print(f"\n✅ 语义可视化已保存到: {output_path}")

    return output_path


if __name__ == "__main__":
    import subprocess

    print("=" * 60)
    print("创建三维语义分割可视化")
    print("=" * 60)

    output_path = create_semantic_visualization()

    print()
    print("=" * 60)
    print("✅ 可视化已创建！")
    print("=" * 60)
    print()
    print("打开以下链接查看三维语义分割结果：")
    print()
    print(f"  file://{os.path.abspath(output_path)}")
    print()
    print("或者在浏览器中打开：")
    print(f"  http://localhost:8000/{os.path.basename(output_path)}")
    print()
    print("操作说明：")
    print("  - 鼠标拖拽：旋转3D场景")
    print("  - 滚轮：缩放")
    print("  - 右键拖拽：平移")
    print()
    print("颜色说明：")
    print("  - 每个点的颜色代表其语义类别")
    print("  - 不是物体的原始RGB颜色")
    print("  - 而是根据语义分割结果分配的颜色")

    # 在浏览器中打开
    print("\n正在打开浏览器...")
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_path)}")
        webbrowser.open(f"http://localhost:8000/{os.path.basename(output_path)}")
    except:
        pass
