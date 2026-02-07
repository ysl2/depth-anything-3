#!/usr/bin/env python3
"""
简化测试脚本 - 跳过回路闭合，只测试核心DA3 + FastSAM功能
"""
import os
import sys

# 添加src目录到路径
sys.path.insert(0, 'src')

import gc
import glob
import json
import tempfile
import shutil

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

# 导入语义分割模块
sys.path.insert(0, 'da3_streaming')
from semantic_segmentation_ultralytics import FastSAMSegmenter, segment_with_memory_cleanup


def load_and_preprocess_images(image_paths, target_size=504):
    """加载和预处理图像"""
    from PIL import Image
    import torchvision.transforms as transforms

    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        # 调整大小
        img = img.resize((target_size, target_size), Image.Resampling.BILINEAR)
        images.append(np.array(img))
    return images


def simple_test():
    """简化测试：加载DA3模型，对几张图像进行推理"""

    # 测试图像目录
    image_dir = "/home/songliyu/Templates/south-building/images"
    output_dir = "./simple_test_output"
    os.makedirs(output_dir, exist_ok=True)

    # 获取测试图像（只用8张）
    images = sorted(
        glob.glob(os.path.join(image_dir, "*.JPG"))[:8]
    )

    if len(images) == 0:
        print(f"No images found in {image_dir}")
        return

    print(f"Using {len(images)} images for testing")

    print("=" * 60)
    print("Step 1: Loading DA3 Model")
    print("=" * 60)

    try:
        # 加载DA3模型
        print("Loading DA3-LARGE from HuggingFace (this may take a few minutes)...")
        model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
        model = model.to("cuda")
        model.eval()
        print("DA3 model loaded successfully!")
    except Exception as e:
        print(f"Failed to load DA3 model: {e}")
        return

    print()
    print("=" * 60)
    print("Step 2: Running DA3 Inference")
    print("=" * 60)

    try:
        # 预处理图像
        print("Preprocessing images...")
        processed_images = load_and_preprocess_images(images)
        print(f"Preprocessed {len(processed_images)} images")

        # 运行DA3推理
        print("Running DA3 inference...")
        torch.cuda.empty_cache()

        with torch.no_grad():
            prediction = model.inference(
                processed_images,
                ref_view_strategy="first",  # 简单策略
                use_ray_pose=False,  # 节省显存
            )

        print(f"DA3 inference completed!")
        print(f"  - Depth shape: {prediction.depth.shape}")
        print(f"  - Conf shape: {prediction.conf.shape}")
        print(f"  - Extrinsics shape: {prediction.extrinsics.shape}")
        print(f"  - Intrinsics shape: {prediction.intrinsics.shape}")

        # 保存DA3结果
        np.savez(
            os.path.join(output_dir, "da3_prediction.npz"),
            depth=prediction.depth,
            conf=prediction.conf,
            extrinsics=prediction.extrinsics,
            intrinsics=prediction.intrinsics,
            images=prediction.processed_images,
        )
        print("DA3 results saved!")

    except Exception as e:
        print(f"DA3 inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("=" * 60)
    print("Step 3: Running FastSAM Segmentation")
    print("=" * 60)

    # 清除DA3显存
    torch.cuda.empty_cache()
    del model
    gc.collect()

    try:
        # 初始化FastSAM
        print("Initializing FastSAM...")
        segmenter = FastSAMSegmenter(
            model_path="./da3_streaming/weights/FastSAM-x.pt",
            device="cuda",
        )

        # 运行语义分割
        print("Running FastSAM segmentation...")
        semantic_results = segment_with_memory_cleanup(
            segmenter,
            prediction.processed_images,
            text_prompts=None,  # 自动模式
        )

        print(f"FastSAM segmentation completed!")
        print(f"  - Semantic masks shape: {semantic_results['masks'].shape}")

        # 保存语义结果
        np.save(
            os.path.join(output_dir, "semantic_masks.npy"),
            semantic_results['masks']
        )
        print("Semantic masks saved!")

    except Exception as e:
        print(f"FastSAM segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("=" * 60)
    print("Step 4: Generating Semantic Point Cloud")
    print("=" * 60)

    try:
        # 生成点云
        from da3_streaming import depth_to_point_cloud_vectorized
        from semantic_ply import save_ply_with_semantics

        print("Generating point cloud...")
        points = depth_to_point_cloud_vectorized(
            prediction.depth,
            prediction.intrinsics,
            prediction.extrinsics
        )

        print(f"Point cloud shape: {points.shape}")

        # 保存点云
        colors = prediction.processed_images
        semantics = semantic_results['masks']

        # 生成语义颜色
        colormap = segmenter.get_semantic_colormap()
        semantic_colors = colormap[semantics]

        # 合并所有点的颜色和语义
        all_points = []
        all_colors = []
        all_sems = []
        all_sem_colors = []

        for i in range(len(points)):
            pts = points[i].reshape(-1, 3)
            cls = colors[i].reshape(-1, 3)
            sems = semantics[i].reshape(-1)
            sem_cls = semantic_colors[i].reshape(-1, 3)

            # 过滤有效点
            conf = prediction.conf[i].reshape(-1)
            mask = conf > (np.mean(conf) * 0.5)

            all_points.append(pts[mask])
            all_colors.append(cls[mask])
            all_sems.append(sems[mask])
            all_sem_colors.append(sem_cls[mask])

        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
        all_sems = np.concatenate(all_sems, axis=0)
        all_sem_colors = np.concatenate(all_sem_colors, axis=0).astype(np.uint8)

        print(f"Total points after filtering: {len(all_points)}")

        # 保存点云
        ply_path = os.path.join(output_dir, "semantic_pointcloud.ply")
        save_ply_with_semantics(
            all_points,
            all_colors,
            ply_path,
            semantics=all_sems,
            semantic_colors=all_sem_colors,
        )
        print(f"Semantic point cloud saved to: {ply_path}")

    except Exception as e:
        print(f"Point cloud generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("=" * 60)
    print("Test Completed Successfully!")
    print("=" * 60)
    print(f"Output saved to: {output_dir}")
    print()
    print("Generated files:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size/1024/1024:.2f} MB)")

    print()
    print("=" * 60)
    print("Visualization Instructions")
    print("=" * 60)
    print("To visualize the semantic point cloud, you can use:")
    print("1. CloudCompare (free, open-source)")
    print("2. MeshLab")
    print("3. Python with trimesh:")
    print("   python -c \"import trimesh; pc = trimesh.load('simple_test_output/semantic_pointcloud.ply'); print(pc.vertices.shape)\"")


if __name__ == "__main__":
    simple_test()
