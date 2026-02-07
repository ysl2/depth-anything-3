#!/usr/bin/env python3
"""
重新生成真正的语义分割可视化 - 使用COCO 80类物体检测
"""
import os
import sys
import gc
import numpy as np
import torch

# 添加src目录到路径
sys.path.insert(0, 'src')

def create_true_semantic_visualization():
    """使用FastSAM检测的物体类别创建语义可视化"""

    # 重新运行FastSAM获取实际检测类别
    sys.path.insert(0, 'da3_streaming')
    from semantic_segmentation_ultralytics import FastSAMSegmenter
    from depth_anything_3.api import DepthAnything3
    from PIL import Image

    print("=" * 60)
    print("重新生成语义分割可视化")
    print("=" * 60)

    # 加载图像
    image_dir = "/home/songliyu/Templates/south-building/images"
    images = sorted([
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.endswith('.JPG')
    ])[:8]

    processed_images = []
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((504, 504), Image.Resampling.BILINEAR)
        processed_images.append(np.array(img))

    # 加载DA3模型获取深度和相机参数
    print("加载DA3模型...")
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

    # 使用FastSAM进行物体检测和语义分割
    print("运行FastSAM物体检测...")
    segmenter = FastSAMSegmenter(
        model_path="./da3_streaming/weights/FastSAM-x.pt",
        device="cuda",
    )
    segmenter.load_model()

    # 对每张图像进行检测
    all_points = []
    all_colors = []
    all_categories = []

    # COCO类别颜色映射 (Cityscapes风格扩展)
    category_colors = {
        0: [0, 0, 0],           # 背景
        1: [128, 64, 128],       # 建筑
        2: [128, 64, 128],       # 墙
        3: [128, 128, 128],     # 地板
        4: [128, 64, 128],       # 椅被
        5: [128, 128, 0],       # 天花板
        10: [128, 64, 128],      # 天空
        11: [220, 20, 60],       # 人
        13: [70, 70, 70],        # 植物
        15: [70, 130, 180],      # 天空2
        16: [70, 130, 180],      # 人2
        58: [70, 130, 180],      # 汽车
        60: [70, 130, 180],       # 其他物体
    }

    print(f"处理 {len(processed_images)} 张图像...")

    # 处理每张图像
    for img_idx, img_array in enumerate(processed_images):
        print(f"  图像 {img_idx + 1}...")

        # 获取深度和相机参数
        depth = prediction.depth[img_idx]
        intrinsics = prediction.intrinsics[img_idx]
        extrinsics = prediction.extrinsics[img_idx]

        # 运行FastSAM
        results = segmenter.model(
            img_array,
            device="cuda",
            imgsz=1024,
            conf=0.4,
            iou=0.9,
        )

        # 创建语义掩码
        H, W = img_array.shape[:2]
        semantic_mask = np.zeros((H, W), dtype=np.uint8)

        # 处理检测结果
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            masks = results[0].masks

            # 获取类别ID
            cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else None

            # 打印检测到的类别
            if cls_ids is not None:
                unique_cls = np.unique(cls_ids)
                print(f"    检测到的COCO类别ID: {unique_cls}")

            if masks is not None:
                for j, mask in enumerate(masks):
                    if mask is not None:
                        mask_np = mask.data.cpu().numpy()
                        if mask_np.ndim == 3:
                            mask_np = mask_np[0]

                        # 调整大小
                        if mask_np.shape != (H, W):
                            mask_np = np.array(
                                Image.fromarray(mask_np.astype(np.uint8))
                                .resize((W, H), Image.NEAREST)
                            )

                        # 获取类别ID
                        if cls_ids is not None and j < len(cls_ids):
                            class_id = int(cls_ids[j]) + 1  # +1 因为0是背景
                        else:
                            class_id = 1  # 默认物体类别

                        # 合并掩码
                        semantic_mask[mask_np > 0] = class_id

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
        cam_coords = cam_coords * depth_flat[:, np.newaxis]  # 广播乘法
        cam_coords_h = np.hstack([cam_coords, np.ones((H*W, 1))])

        # 相机坐标 -> 世界坐标
        c2w = np.eye(4)
        c2w[:3, :4] = extrinsics
        c2w_inv = np.linalg.inv(c2w)
        world_coords_h = (c2w_inv @ cam_coords_h.T).T
        points = world_coords_h[:, :3]

        # 过滤有效点
        valid = (points[:, 2] > 0) & np.isfinite(points).all(axis=1)

        # 获取语义标签
        sem_flat = semantic_mask.flatten()
        sem_colors = np.array([category_colors.get(s, [128, 128, 128]) for s in sem_flat])

        all_points.append(points[valid])
        all_colors.append(sem_colors[valid])
        all_categories.append(sem_flat[valid])

        print(f"    检测到 {len(cls_ids) if cls_ids is not None else 0} 个物体类别")
        print(f"    有效点数: {np.sum(valid)}")

    segmenter.unload_model()

    # 合并所有点
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
    all_categories = np.concatenate(all_categories, axis=0)

    print(f"\n总点数: {len(all_points)}")
    print(f"唯一类别: {np.unique(all_categories)}")

    # 统计类别分布
    unique_cats, counts = np.unique(all_categories, return_counts=True)
    print("\n类别分布:")
    for cat, count in sorted(zip(unique_cats, counts), key=lambda x: -x[1]):
        pct = count / len(all_categories) * 100
        color = category_colors.get(cat, [128, 128, 128])
        print(f"  类别 {cat:3d}: {count:8,} 点 ({pct:5.2f}%) - RGB{tuple(color)}")

    # 采样并创建可视化
    import plotly.graph_objects as go

    sample_size = min(150000, len(all_points))
    indices = np.random.choice(len(all_points), sample_size, replace=False)

    points_sampled = all_points[indices]
    colors_sampled = all_colors[indices] / 255.0
    categories_sampled = all_categories[indices]

    # 创建3D图
    fig = go.Figure(data=[go.Scatter3d(
        x=points_sampled[:, 0],
        y=points_sampled[:, 1],
        z=points_sampled[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=colors_sampled,
            opacity=0.7,
            line=dict(width=0)
        ),
        text=[f'类别 {cat}' for cat in categories_sampled]
    )])

    # 根据类别分布生成标题
    cat_names = {
        0: '背景',
        1: '建筑/墙',
        10: '天空',
        11: '人',
        13: '植物',
        58: '汽车',
        60: '其他'
    }

    # 构建类别描述
    cat_descriptions = []
    for cat, count in sorted(zip(unique_cats, counts), key=lambda x: -x[1]):
        pct = count / len(all_categories) * 100
        name = cat_names.get(cat, f'类别{cat}')
        color = category_colors.get(cat, [128, 128, 128])
        cat_descriptions.append(f"{name} ({pct:.1f}%)")

    title = '三维语义分割可视化 - ' + ', '.join(cat_descriptions[:5])

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
        margin=dict(t=50),
        paper_bgcolor='rgb(10,10,10)',
        font=dict(color='white')
    )

    # 保存
    output_path = "./semantic_test_output/true_semantic_3d.html"
    fig.write_html(output_path)

    print(f"\n✅ 真正的语义分割可视化已保存: {output_path}")

    # 在浏览器中打开
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:8000/true_semantic_3d.html")
    except:
        pass

    return output_path


if __name__ == "__main__":
    create_true_semantic_visualization()
