#!/usr/bin/env python3
"""
优化的三维重建脚本

改进点：
1. 使用更多图像 (64张而不是8张)
2. 使用use_ray_pose=True获得更准确的位姿
3. 使用更好的参考帧选择策略
4. 使用更大的chunk_size提高处理效率
5. 添加语义分割输出
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


def improved_3d_reconstruction():
    """改进的三维重建"""

    print("=" * 70)
    print("优化的三维重建 + 语义分割")
    print("=" * 70)

    # 配置
    image_dir = "/home/songliyu/Templates/south-building/images"
    num_images = 64  # 使用64张图像
    output_dir = "./semantic_test_output"

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载图像
    print(f"\n[1/6] 加载图像...")
    images = sorted([
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.endswith('.JPG')
    ])[:num_images]

    print(f"  已加载 {len(images)} 张图像")

    processed_images = []
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((504, 504), Image.Resampling.BILINEAR)
        processed_images.append(np.array(img))

    print(f"  图像处理完成，尺寸: {processed_images[0].shape}")

    # 2. 运行DA3获取深度和相机参数 (使用优化参数)
    print(f"\n[2/6] 运行DA3获取深度和位姿...")
    print(f"  参数: use_ray_pose=True, ref_view_strategy=saddle_balanced")

    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to("cuda")

    with torch.no_grad():
        prediction = model.inference(
            processed_images,
            ref_view_strategy="saddle_balanced",  # 更好的参考帧选择
            use_ray_pose=True,  # 使用ray-based位姿估计（更准确但更慢）
        )

    # 清除DA3显存
    torch.cuda.empty_cache()
    del model
    gc.collect()
    print("  DA3推理完成")

    # 3. 运行SegFormer语义分割
    print(f"\n[3/6] 运行SegFormer语义分割...")
    segmenter = SegFormerSegmenter(
        model_name="nvidia/segformer-b5-finetuned-ade-640-640",
        device="cuda"
    )
    segmenter.load_model()

    semantic_results = segmenter.segment_images(processed_images)

    # 打印第一张和中间一张的类别统计
    print("\n  图像1的语义类别:")
    segmenter.print_class_statistics(semantic_results[0]['masks'])

    if len(semantic_results) > 30:
        mid_idx = len(semantic_results) // 2
        print(f"\n  图像{mid_idx+1}的语义类别:")
        segmenter.print_class_statistics(semantic_results[mid_idx]['masks'])

    # 4. 生成3D语义点云
    print(f"\n[4/6] 生成3D语义点云...")

    all_points = []
    all_colors = []
    all_categories = []

    for img_idx in range(len(processed_images)):
        if (img_idx + 1) % 10 == 0:
            print(f"  处理进度: {img_idx+1}/{len(processed_images)}...")

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

    # 5. 分析点云质量
    print(f"\n[5/6] 分析点云质量...")

    # 分析点云范围
    xyz_min = all_points.min(axis=0)
    xyz_max = all_points.max(axis=0)
    xyz_range = xyz_max - xyz_min

    print(f"  点云范围:")
    print(f"    X: [{xyz_min[0]:.2f}, {xyz_max[0]:.2f}] 跨度={xyz_range[0]:.2f}m")
    print(f"    Y: [{xyz_min[1]:.2f}, {xyz_max[1]:.2f}] 跨度={xyz_range[1]:.2f}m")
    print(f"    Z: [{xyz_min[2]:.2f}, {xyz_max[2]:.2f}] 跨度={xyz_range[2]:.2f}m")

    # 统计类别分布
    unique_cats, counts = np.unique(all_categories, return_counts=True)

    print("\n  主要语义类别分布 (前15个):")
    for cat, count in sorted(zip(unique_cats, counts), key=lambda x: -x[1])[:15]:
        pct = count / len(all_categories) * 100
        name = segmenter.get_class_name(int(cat))
        color = segmenter.get_class_color(int(cat))
        print(f"    {name:20s} (ID={cat:3d}): {count:8,} 点 ({pct:5.2f}%) - RGB{tuple(color)}")

    # 6. 创建可视化
    print(f"\n[6/6] 创建3D可视化...")

    # 采样 (采样更多点以获得更好的可视化效果)
    sample_size = min(500000, len(all_points))
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
            size=1.5,  # 更小的点以显示更多细节
            color=colors_sampled,
            opacity=0.6,
            line=dict(width=0)
        ),
        text=hover_texts,
        hoverinfo='text+x+y+z'
    )])

    # 构建标题
    cat_descriptions = []
    for cat, count in sorted(zip(unique_cats, counts), key=lambda x: -x[1])[:6]:
        pct = count / len(all_categories) * 100
        name = segmenter.get_class_name(int(cat))
        cat_descriptions.append(f"{name} ({pct:.1f}%)")

    title = f'优化重建 + ADE20K语义分割 - {len(images)}张图像 - ' + ', '.join(cat_descriptions)

    # 设置相机视角以更好地展示场景
    center_x = (xyz_min[0] + xyz_max[0]) / 2
    center_y = (xyz_min[1] + xyz_max[1]) / 2
    center_z = (xyz_min[2] + xyz_max[2]) / 2
    max_range = max(xyz_range) * 2

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', range=[xyz_min[0], xyz_max[0]]),
            yaxis=dict(title='Y', range=[xyz_min[1], xyz_max[1]]),
            zaxis=dict(title='Z', range=[xyz_min[2], xyz_max[2]]),
            bgcolor='rgb(10,10,10)',
            camera=dict(
                eye=dict(x=max_range, y=max_range, z=max_range*0.5),
                center=dict(x=0, y=0, z=0)
            )
        ),
        height=900,
        margin=dict(t=100, l=0, r=0, b=0),
        paper_bgcolor='rgb(10,10,10)',
        font=dict(color='white', size=10)
    )

    # 保存
    output_path = os.path.join(output_dir, "improved_reconstruction_64img.html")
    fig.write_html(output_path)

    # 清理
    segmenter.unload_model()

    print(f"\n{'=' * 70}")
    print(f"✅ 优化的三维重建完成！")
    print(f"{'=' * 70}")
    print(f"\n使用图像数: {len(images)}")
    print(f"点云范围: {xyz_range[0]:.2f}m × {xyz_range[1]:.2f}m × {xyz_range[2]:.2f}m")
    print(f"总点数: {len(all_points):,}")
    print(f"可视化已保存: {output_path}")
    print(f"请在浏览器中打开查看: http://localhost:8000/improved_reconstruction_64img.html")
    print()

    return output_path, all_points, all_colors, all_categories


if __name__ == "__main__":
    import subprocess
    import time

    # 启动HTTP服务器
    subprocess.Popen([
        'python', '-m', 'http.server', '8000',
        '--directory', './semantic_test_output'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)

    # 运行重建
    output_path, points, colors, categories = improved_3d_reconstruction()

    print("=" * 70)
    print("重建完成！请打开浏览器查看结果")
    print("=" * 70)
