#!/usr/bin/env python3
"""
使用YOLOv8进行真正的80类COCO物体检测和语义分割
"""
import os
import sys
import gc
import numpy as np
import torch
from PIL import Image

# 添加src目录到路径
sys.path.insert(0, 'src')

def create_true_semantic_with_yolo():
    """使用YOLOv8进行80类COCO物体检测"""

    from depth_anything_3.api import DepthAnything3

    print("=" * 60)
    print("使用YOLOv8进行COCO 80类检测")
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

    # 加载YOLOv8模型
    print("加载YOLOv8模型...")
    from ultralytics import YOLO
    yolo_model = YOLO('yolov8x.pt')  # 使用最大的YOLOv8模型

    # COCO 80类颜色映射 (Cityscapes风格)
    coco_colors = {
        0: [220, 20, 60],      # person - 红色
        1: [119, 11, 32],      # bicycle - 深红
        2: [0, 0, 142],        # car - 蓝色
        3: [0, 0, 70],         # motorcycle - 深蓝
        4: [0, 60, 100],       # airplane
        5: [0, 80, 100],       # bus
        6: [0, 0, 230],        # train
        7: [140, 140, 140],    # truck - 灰色
        8: [70, 70, 70],       # boat
        9: [190, 153, 153],    # traffic light
        10: [250, 170, 30],    # fire hydrant
        11: [220, 220, 0],     # stop sign
        13: [107, 142, 35],    # bench - 绿色
        15: [70, 130, 180],    # bird - 天蓝
        16: [220, 20, 60],     # cat
        17: [220, 20, 60],     # dog
        18: [220, 20, 60],     # horse
        19: [220, 20, 60],     # sheep
        20: [220, 20, 60],     # cow
        27: [107, 142, 35],    # potted plant
        28: [128, 64, 128],    # bed
        31: [152, 251, 152],   # couch
        32: [70, 70, 70],      # tv
        33: [70, 70, 70],      # laptop
        56: [70, 130, 180],    # chair
        57: [70, 70, 70],      # dining table
        58: [152, 251, 152],   # toilet
        59: [0, 0, 142],       # remote
        60: [0, 0, 142],       # cell phone
        62: [128, 64, 128],    # microwave
        63: [128, 64, 128],    # oven
        64: [0, 0, 142],       # toaster
        65: [70, 130, 180],    # sink
        66: [128, 64, 128],    # refrigerator
        72: [70, 130, 180],    # book
    }

    # COCO类别名称
    coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    all_points = []
    all_colors = []
    all_categories = []
    all_class_names = []

    print(f"处理 {len(processed_images)} 张图像...")

    # 处理每张图像
    for img_idx, img_array in enumerate(processed_images):
        print(f"  图像 {img_idx + 1}...")

        # 获取深度和相机参数
        depth = prediction.depth[img_idx]
        intrinsics = prediction.intrinsics[img_idx]
        extrinsics = prediction.extrinsics[img_idx]

        # 运行YOLOv8检测
        results = yolo_model(
            img_array,
            device="cuda",
            imgsz=640,
            conf=0.25,
            iou=0.45,
            verbose=False
        )

        # 创建语义掩码
        H, W = img_array.shape[:2]
        semantic_mask = np.zeros((H, W), dtype=np.uint8)

        # 收集检测到的类别
        detected_classes = []

        # 处理检测结果
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            cls_ids = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else None
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else None

            if cls_ids is not None and xyxy is not None:
                for j, (cls_id, box) in enumerate(zip(cls_ids, xyxy)):
                    cls_id = int(cls_id)
                    detected_classes.append(cls_id)

                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.astype(int)

                    # 限制在图像范围内
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)

                    # 标记该区域为对应的类别ID (使用+1以保留0为背景)
                    semantic_mask[y1:y2, x1:x2] = cls_id + 1

        print(f"    检测到类别: {set(detected_classes)}")

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

        # 获取语义标签
        sem_flat = semantic_mask.flatten()

        # 获取颜色和类别名称
        sem_colors = np.zeros((len(sem_flat), 3), dtype=np.uint8)
        sem_class_names = []

        for i, s in enumerate(sem_flat):
            if s == 0:  # 背景
                sem_colors[i] = [70, 70, 70]  # 灰色
                sem_class_names.append('background')
            else:
                cls_id = s - 1  # 减1恢复原始类别ID
                color = coco_colors.get(cls_id, [128, 128, 128])
                sem_colors[i] = color
                if cls_id < len(coco_names):
                    sem_class_names.append(coco_names[cls_id])
                else:
                    sem_class_names.append(f'class_{cls_id}')

        all_points.append(points[valid])
        all_colors.append(sem_colors[valid])
        all_categories.append(sem_flat[valid])
        all_class_names.extend([sem_class_names[i] for i in range(len(sem_flat)) if valid[i]])

        print(f"    有效点数: {np.sum(valid)}")

    # 清理YOLO模型
    del yolo_model
    torch.cuda.empty_cache()
    gc.collect()

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
        if cat == 0:
            name = 'background'
            color = [70, 70, 70]
        else:
            cls_id = cat - 1
            if cls_id < len(coco_names):
                name = coco_names[cls_id]
            else:
                name = f'class_{cls_id}'
            color = coco_colors.get(cls_id, [128, 128, 128])
        print(f"  {name:20s}: {count:8,} 点 ({pct:5.2f}%) - RGB{tuple(color)}")

    # 采样并创建可视化
    import plotly.graph_objects as go

    sample_size = min(200000, len(all_points))
    indices = np.random.choice(len(all_points), sample_size, replace=False)

    points_sampled = all_points[indices]
    colors_sampled = all_colors[indices] / 255.0
    categories_sampled = all_categories[indices]

    # 创建hover文本
    hover_texts = []
    for cat in categories_sampled:
        if cat == 0:
            hover_texts.append('Background')
        else:
            cls_id = int(cat - 1)
            if cls_id < len(coco_names):
                hover_texts.append(coco_names[cls_id])
            else:
                hover_texts.append(f'Class {cls_id}')

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
        text=hover_texts,
        hoverinfo='text+x+y+z'
    )])

    # 构建类别描述
    cat_descriptions = []
    for cat, count in sorted(zip(unique_cats, counts), key=lambda x: -x[1]):
        pct = count / len(all_categories) * 100
        if cat == 0:
            name = 'background'
        else:
            cls_id = cat - 1
            name = coco_names[cls_id] if cls_id < len(coco_names) else f'class_{cls_id}'
        cat_descriptions.append(f"{name} ({pct:.1f}%)")

    title = 'COCO 80类语义分割 - ' + ', '.join(cat_descriptions[:5])

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
        margin=dict(t=80),
        paper_bgcolor='rgb(10,10,10)',
        font=dict(color='white')
    )

    # 保存
    output_path = "./semantic_test_output/coco_80class_semantic.html"
    fig.write_html(output_path)

    print(f"\n✅ COCO 80类语义分割可视化已保存: {output_path}")

    # 在浏览器中打开
    try:
        import webbrowser
        import subprocess
        # 启动HTTP服务器
        subprocess.Popen(['python', '-m', 'http.server', '8000', '--directory', './semantic_test_output'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        webbrowser.open("http://localhost:8000/coco_80class_semantic.html")
    except:
        pass

    return output_path


if __name__ == "__main__":
    create_true_semantic_with_yolo()
