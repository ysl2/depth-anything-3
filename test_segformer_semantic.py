#!/usr/bin/env python3
"""
SegFormer语义分割测试脚本

使用SegFormer进行ADE20K 150类语义分割，支持建筑、墙壁、地面、天空、植被等。
"""
import os
import sys
import gc
import numpy as np
import torch
from PIL import Image
import plotly.graph_objects as go

# 添加路径
sys.path.insert(0, 'src')
sys.path.insert(0, 'da3_streaming')

from depth_anything_3.api import DepthAnything3
from semantic_segmentation_segformer import SegFormerSegmenter


def test_segformer_semantic():
    """测试SegFormer语义分割"""

    print("=" * 70)
    print("SegFormer ADE20K 150类语义分割测试")
    print("=" * 70)

    # 配置
    image_dir = "/home/songliyu/Templates/south-building/images"
    num_images = 8
    output_dir = "./semantic_test_output"

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载图像
    print(f"\n[1/5] 加载图像...")
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

    # 2. 运行DA3获取深度和相机参数
    print(f"\n[2/5] 运行DA3获取深度和位姿...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to("cuda")

    with torch.no_grad():
        prediction = model.inference(
            processed_images,
            ref_view_strategy="first",
            use_ray_pose=False,
        )

    # 清除DA3显存
    torch.cuda.empty_cache()
    del model
    gc.collect()
    print("  DA3推理完成")

    # 3. 运行SegFormer语义分割
    print(f"\n[3/5] 运行SegFormer语义分割...")
    segmenter = SegFormerSegmenter(
        model_name="nvidia/segformer-b5-finetuned-ade-640-640",
        device="cuda"
    )
    segmenter.load_model()

    semantic_results = segmenter.segment_images(processed_images)

    # 打印第一张图像的类别统计
    print("\n  图像1的语义类别:")
    segmenter.print_class_statistics(semantic_results[0]['masks'])

    # 4. 生成3D语义点云
    print(f"\n[4/5] 生成3D语义点云...")

    all_points = []
    all_colors = []
    all_categories = []
    all_class_names = []

    for img_idx in range(len(processed_images)):
        # 获取深度和相机参数
        depth = prediction.depth[img_idx]
        intrinsics = prediction.intrinsics[img_idx]
        extrinsics = prediction.extrinsics[img_idx]
        semantic_mask = semantic_results[img_idx]['masks']
        class_colors = semantic_results[img_idx]['class_colors']

        # 生成点云
        H, W = depth.shape
        us, vs = np.meshgrid(np.arange(W), np.arange(H))

        # 相机内参
        K = intrinsics
        K_inv = np.linalg.inv(K)

        # 像素坐标 -> 相机坐标
        coords = np.stack([us.flatten(), vs.flatten(), np.ones(H*W)], axis=0)
        cam_coords = (K_inv @ coords).T
        depth_flat = depth.flatten()
        cam_coords = cam_coords * depth_flat[:, np.newaxis]
        cam_coords_h = np.hstack([cam_coords, np.ones((H*W, 1))])

        # 相机坐标 -> 世界坐标
        c2w = np.eye(4)
        c2w[:3, :4] = extrinsics
        c2w_inv = np.linalg.inv(c2w)
        world_coords_h = (c2w_inv @ cam_coords_h.T).T
        points = world_coords_h[:, :3]

        # 过滤有效点
        valid = (points[:, 2] > 0) & np.isfinite(points).all(axis=1)

        # 获取语义标签和颜色
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

    # 5. 创建可视化
    print(f"\n[5/5] 创建3D可视化...")

    # 统计类别分布
    unique_cats, counts = np.unique(all_categories, return_counts=True)

    print("\n  主要语义类别分布 (前15个):")
    for cat, count in sorted(zip(unique_cats, counts), key=lambda x: -x[1])[:15]:
        pct = count / len(all_categories) * 100
        name = segmenter.get_class_name(int(cat))
        color = segmenter.get_class_color(int(cat))
        print(f"    {name:20s} (ID={cat:3d}): {count:8,} 点 ({pct:5.2f}%) - RGB{tuple(color)}")

    # 采样
    sample_size = min(300000, len(all_points))
    indices = np.random.choice(len(all_points), sample_size, replace=False)

    points_sampled = all_points[indices]
    colors_sampled = all_colors[indices] / 255.0
    categories_sampled = all_categories[indices]

    # 创建hover文本
    hover_texts = [segmenter.get_class_name(int(cat)) for cat in categories_sampled]

    # 创建3D图
    fig = go.Figure(data=[go.Scatter3d(
        x=points_sampled[:, 0],
        y=points_sampled[:, 1],
        z=points_sampled[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors_sampled,
            opacity=0.7,
            line=dict(width=0)
        ),
        text=hover_texts,
        hoverinfo='text+x+y+z'
    )])

    # 构建标题
    cat_descriptions = []
    for cat, count in sorted(zip(unique_cats, counts), key=lambda x: -x[1])[:8]:
        pct = count / len(all_categories) * 100
        name = segmenter.get_class_name(int(cat))
        cat_descriptions.append(f"{name} ({pct:.1f}%)")

    title = 'ADE20K 150类语义分割 - ' + ', '.join(cat_descriptions)

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            bgcolor='rgb(10,10,10)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=800,
        margin=dict(t=100),
        paper_bgcolor='rgb(10,10,10)',
        font=dict(color='white', size=10)
    )

    # 保存
    output_path = os.path.join(output_dir, "segformer_ade20k_semantic.html")
    fig.write_html(output_path)

    # 清理
    segmenter.unload_model()

    print(f"\n{'=' * 70}")
    print(f"✅ SegFormer语义分割完成！")
    print(f"{'=' * 70}")
    print(f"\n可视化已保存: {output_path}")
    print(f"请在浏览器中打开查看: http://localhost:8000/segformer_ade20k_semantic.html")
    print()

    return output_path


if __name__ == "__main__":
    import subprocess

    # 启动HTTP服务器
    subprocess.Popen([
        'python', '-m', 'http.server', '8000',
        '--directory', './semantic_test_output'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 运行测试
    test_segformer_semantic()

    print("=" * 70)
    print("测试完成！请打开浏览器查看结果")
    print("=" * 70)
