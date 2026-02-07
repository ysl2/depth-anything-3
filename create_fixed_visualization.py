#!/usr/bin/env python3
"""
创建修复的3D可视化 - 解决空显示问题
"""
import os
import sys
import gc
import numpy as np
import torch
from PIL import Image
import plotly.graph_objects as go

sys.path.insert(0, 'src')
sys.path.insert(0, 'da3_streaming')

from depth_anything_3.api import DepthAnything3
from semantic_segmentation_segformer import SegFormerSegmenter


def create_fixed_visualization():
    """创建修复的3D可视化"""

    print("=" * 70)
    print("创建修复的3D可视化")
    print("=" * 70)

    # 配置
    image_dir = "/home/songliyu/Templates/south-building/images"
    num_images = 32  # 使用32张图像
    output_dir = "./semantic_test_output"

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载图像
    print(f"\n[1/4] 加载图像...")
    images = sorted([
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.endswith('.JPG')
    ])[:num_images]

    processed_images = []
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((504, 504), Image.Resampling.BILINEAR)
        processed_images.append(np.array(img))

    print(f"  已加载 {len(processed_images)} 张图像")

    # 2. 运行DA3
    print(f"\n[2/4] 运行DA3...")

    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to("cuda")

    with torch.no_grad():
        prediction = model.inference(
            processed_images,
            ref_view_strategy="saddle_balanced",
            use_ray_pose=True,
        )

    torch.cuda.empty_cache()
    del model
    gc.collect()

    # 3. 运行语义分割
    print(f"\n[3/4] 运行SegFormer...")
    segmenter = SegFormerSegmenter(
        model_name="nvidia/segformer-b5-finetuned-ade-640-640",
        device="cuda"
    )
    segmenter.load_model()

    semantic_results = segmenter.segment_images(processed_images)

    # 4. 生成点云
    print(f"\n[4/4] 生成点云并创建可视化...")

    all_points = []
    all_colors = []
    all_categories = []

    for img_idx in range(len(processed_images)):
        if (img_idx + 1) % 8 == 0:
            print(f"  处理进度: {img_idx+1}/{len(processed_images)}...")

        depth = prediction.depth[img_idx]
        intrinsics = prediction.intrinsics[img_idx]
        extrinsics = prediction.extrinsics[img_idx]
        semantic_mask = semantic_results[img_idx]['masks']
        class_colors = semantic_results[img_idx]['class_colors']

        H, W = depth.shape
        us, vs = np.meshgrid(np.arange(W), np.arange(H))

        K = intrinsics
        K_inv = np.linalg.inv(K)

        coords = np.stack([us.flatten(), vs.flatten(), np.ones(H*W)], axis=0)
        cam_coords = (K_inv @ coords).T
        depth_flat = depth.flatten()
        cam_coords = cam_coords * depth_flat[:, np.newaxis]
        cam_coords_h = np.hstack([cam_coords, np.ones((H*W, 1))])

        c2w = np.eye(4)
        c2w[:3, :4] = extrinsics
        c2w_inv = np.linalg.inv(c2w)
        world_coords_h = (c2w_inv @ cam_coords_h.T).T
        points = world_coords_h[:, :3]

        valid = (points[:, 2] > 0) & np.isfinite(points).all(axis=1)

        sem_flat = semantic_mask.flatten()
        sem_colors = class_colors.reshape(-1, 3)

        all_points.append(points[valid])
        all_colors.append(sem_colors[valid])
        all_categories.append(sem_flat[valid])

    # 合并所有点
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
    all_categories = np.concatenate(all_categories, axis=0)

    print(f"  总点数: {len(all_points):,}")

    # 分析点云
    xyz_min = all_points.min(axis=0)
    xyz_max = all_points.max(axis=0)
    xyz_center = (xyz_min + xyz_max) / 2

    print(f"\n  点云中心: [{xyz_center[0]:.2f}, {xyz_center[1]:.2f}, {xyz_center[2]:.2f}]")
    print(f"  点云范围: X=[{xyz_min[0]:.2f}, {xyz_max[0]:.2f}], "
          f"Y=[{xyz_min[1]:.2f}, {xyz_max[1]:.2f}], "
          f"Z=[{xyz_min[2]:.2f}, {xyz_max[2]:.2f}]")

    # 统计类别
    unique_cats, counts = np.unique(all_categories, return_counts=True)
    print("\n  主要语义类别:")
    for cat, count in sorted(zip(unique_cats, counts), key=lambda x: -x[1])[:10]:
        pct = count / len(all_categories) * 100
        name = segmenter.get_class_name(int(cat))
        print(f"    {name}: {count:,} 点 ({pct:.1f}%)")

    # 采样
    sample_size = min(300000, len(all_points))
    indices = np.random.choice(len(all_points), sample_size, replace=False)

    points_sampled = all_points[indices]
    colors_sampled = all_colors[indices] / 255.0

    # 创建hover文本
    hover_texts = [segmenter.get_class_name(int(cat)) for cat in all_categories[indices]]

    # 创建3D图 - 关键修复：不设置range限制，让plotly自动调整
    fig = go.Figure(data=[go.Scatter3d(
        x=points_sampled[:, 0],
        y=points_sampled[:, 1],
        z=points_sampled[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors_sampled,
            opacity=0.6,
            line=dict(width=0)
        ),
        text=hover_texts,
        hoverinfo='text'
    )])

    # 构建标题
    cat_descriptions = []
    for cat, count in sorted(zip(unique_cats, counts), key=lambda x: -x[1])[:5]:
        pct = count / len(all_categories) * 100
        name = segmenter.get_class_name(int(cat))
        cat_descriptions.append(f"{name} ({pct:.1f}%)")

    title = f'3D重建 + ADE20K语义分割 - {len(images)}张图像<br>' + ', '.join(cat_descriptions)

    # 修复：不设置axis range，设置合适的相机位置
    max_range = max(xyz_max - xyz_min)
    eye_distance = max_range * 2

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', backgroundcolor='rgb(20,20,20)', gridcolor='rgb(40,40,40)'),
            yaxis=dict(title='Y', backgroundcolor='rgb(20,20,20)', gridcolor='rgb(40,40,40)'),
            zaxis=dict(title='Z', backgroundcolor='rgb(20,20,20)', gridcolor='rgb(40,40,40)'),
            bgcolor='rgb(10,10,10)',
            aspectmode='data',  # 关键：使用实际数据比例
            camera=dict(
                eye=dict(x=eye_distance, y=eye_distance, z=eye_distance*0.6),
                center=dict(x=0, y=0, z=0)  # 看向原点
            )
        ),
        height=800,
        margin=dict(t=80, l=0, r=0, b=0),
        paper_bgcolor='rgb(10,10,10)',
        font=dict(color='white', size=11)
    )

    # 保存
    output_path = os.path.join(output_dir, "fixed_visualization.html")
    fig.write_html(output_path)

    segmenter.unload_model()

    print(f"\n{'=' * 70}")
    print(f"✅ 修复的可视化已创建！")
    print(f"{'=' * 70}")
    print(f"\n请在浏览器中打开: http://localhost:8000/fixed_visualization.html")
    print()

    return output_path


if __name__ == "__main__":
    import subprocess
    import time

    subprocess.Popen([
        'python', '-m', 'http.server', '8000',
        '--directory', './semantic_test_output'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)

    create_fixed_visualization()

    print("=" * 70)
    print("可视化已创建！请打开浏览器查看")
    print("=" * 70)
