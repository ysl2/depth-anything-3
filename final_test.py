#!/usr/bin/env python3
"""
最终测试脚本 - 完整的DA3 + FastSAM语义分割测试
"""
import os
import sys

# 添加src目录到路径
sys.path.insert(0, 'src')

import gc
import glob
import tempfile
import shutil

import numpy as np
import torch
from depth_anything_3.api import DepthAnything3

# 导入语义分割模块
sys.path.insert(0, 'da3_streaming')
from semantic_segmentation_ultralytics import FastSAMSegmenter, segment_with_memory_cleanup
from semantic_ply import save_ply_with_semantics


def depth_to_point_cloud(depth, intrinsics, extrinsics):
    """
    将深度图转换为3D点云

    Args:
        depth: (N, H, W) 深度图
        intrinsics: (N, 3, 3) 相机内参
        extrinsics: (N, 3, 4) 相机外参

    Returns:
        points: (N, H, W, 3) 世界坐标点云
    """
    N, H, W = depth.shape

    # 创建像素坐标网格
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    coords = np.stack([us, vs, ones], axis=-1)  # (H, W, 3)

    all_points = []

    for i in range(N):
        # 相机内参
        K = intrinsics[i]
        K_inv = np.linalg.inv(K)

        # 像素坐标转相机坐标
        cam_coords = (K_inv @ coords.reshape(-1, 3).T).T  # (H*W, 3)
        cam_coords = cam_coords * depth[i].reshape(-1, 1)  # (H*W, 3)

        # 齐次坐标
        cam_coords_h = np.hstack([cam_coords, np.ones((len(cam_coords), 1))])

        # 相机坐标转世界坐标
        c2w = np.eye(4)
        c2w[:3, :4] = extrinsics[i]
        c2w_inv = np.linalg.inv(c2w)

        world_coords_h = (c2w_inv @ cam_coords_h.T).T
        world_coords = world_coords_h[:, :3]

        all_points.append(world_coords.reshape(H, W, 3))

    return np.array(all_points)


def final_test():
    """完整测试：DA3深度估计 + FastSAM语义分割 + 点云生成"""

    # 测试图像目录
    image_dir = "/home/songliyu/Templates/south-building/images"
    output_dir = "./semantic_test_output"
    os.makedirs(output_dir, exist_ok=True)

    # 获取测试图像（只用8张）
    images = sorted(glob.glob(os.path.join(image_dir, "*.JPG")))[:8]

    if len(images) == 0:
        print(f"No images found in {image_dir}")
        return

    print(f"Using {len(images)} images for testing")

    # 加载和预处理图像
    from PIL import Image
    processed_images = []
    for img_path in images:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((504, 504), Image.Resampling.BILINEAR)
        processed_images.append(np.array(img))

    print("=" * 70)
    print("Step 1: Loading DA3 Model")
    print("=" * 70)

    try:
        print("Loading DA3-LARGE from HuggingFace...")
        model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
        model = model.to("cuda")
        model.eval()
        print("DA3 model loaded successfully!")
    except Exception as e:
        print(f"Failed to load DA3 model: {e}")
        return

    print()
    print("=" * 70)
    print("Step 2: Running DA3 Inference (Depth + Camera Pose)")
    print("=" * 70)

    try:
        torch.cuda.empty_cache()
        with torch.no_grad():
            prediction = model.inference(
                processed_images,
                ref_view_strategy="first",
                use_ray_pose=False,
            )

        print(f"DA3 inference completed!")
        print(f"  - Depth shape: {prediction.depth.shape}")
        print(f"  - Conf shape: {prediction.conf.shape}")
        print(f"  - Extrinsics shape: {prediction.extrinsics.shape}")
        print(f"  - Intrinsics shape: {prediction.intrinsics.shape}")

    except Exception as e:
        print(f"DA3 inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("=" * 70)
    print("Step 3: Running FastSAM Semantic Segmentation")
    print("=" * 70)

    # 清除DA3显存
    torch.cuda.empty_cache()
    del model
    gc.collect()

    try:
        print("Initializing FastSAM...")
        segmenter = FastSAMSegmenter(
            model_path="./da3_streaming/weights/FastSAM-x.pt",
            device="cuda",
        )

        print("Running FastSAM segmentation...")
        semantic_results = segment_with_memory_cleanup(
            segmenter,
            prediction.processed_images,
        )

        print(f"FastSAM segmentation completed!")
        print(f"  - Semantic masks shape: {semantic_results['masks'].shape}")
        print(f"  - Unique classes: {np.unique(semantic_results['masks'])}")

    except Exception as e:
        print(f"FastSAM segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("=" * 70)
    print("Step 4: Generating Semantic Point Cloud")
    print("=" * 70)

    try:
        print("Converting depth maps to 3D point cloud...")
        points = depth_to_point_cloud(
            prediction.depth,
            prediction.intrinsics,
            prediction.extrinsics
        )

        print(f"Point cloud shape: {points.shape}")

        # 合并所有点并过滤
        all_points = []
        all_colors = []
        all_sems = []
        all_sem_colors = []

        colormap = segmenter.get_semantic_colormap()
        semantics = semantic_results['masks']
        semantic_colors = colormap[semantics]

        for i in range(len(points)):
            pts = points[i].reshape(-1, 3)
            cls = prediction.processed_images[i].reshape(-1, 3)
            sems = semantics[i].reshape(-1)
            sem_cls = semantic_colors[i].reshape(-1, 3)

            # 过滤有效点
            conf = prediction.conf[i].reshape(-1)
            mask = (conf > (np.mean(conf) * 0.5)) & (conf > 0.01)

            all_points.append(pts[mask])
            all_colors.append(cls[mask])
            all_sems.append(sems[mask])
            all_sem_colors.append(sem_cls[mask])

        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
        all_sems = np.concatenate(all_sems, axis=0)
        all_sem_colors = np.concatenate(all_sem_colors, axis=0).astype(np.uint8)

        print(f"Total points after filtering: {len(all_points)}")

        # 保存带语义的点云
        ply_path = os.path.join(output_dir, "semantic_pointcloud.ply")
        save_ply_with_semantics(
            all_points,
            all_colors,
            ply_path,
            semantics=all_sems,
            semantic_colors=all_sem_colors,
        )
        print(f"Semantic point cloud saved to: {ply_path}")

        # 保存语义颜色映射
        np.save(os.path.join(output_dir, "colormap.npy"), colormap)

    except Exception as e:
        print(f"Point cloud generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("=" * 70)
    print("Step 5: Creating Visualization Preview")
    print("=" * 70)

    try:
        # 创建可视化预览图
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 4, figure=fig)

        # 显示原图
        for i in range(min(4, len(processed_images))):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(processed_images[i])
            ax.set_title(f"Image {i+1}")
            ax.axis('off')

        # 显示深度图
        for i in range(min(4, len(prediction.depth))):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(prediction.depth[i], cmap='viridis')
            ax.set_title(f"Depth {i+1}")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        preview_path = os.path.join(output_dir, "preview.png")
        plt.savefig(preview_path, dpi=100)
        plt.close()
        print(f"Preview saved to: {preview_path}")

        # 显示语义掩码
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i in range(min(8, len(semantics))):
            ax = axes[i // 4, i % 4]
            ax.imshow(semantics[i], cmap='tab20')
            ax.set_title(f"Semantic Mask {i+1}")
            ax.axis('off')
        plt.tight_layout()
        semantic_preview_path = os.path.join(output_dir, "semantic_preview.png")
        plt.savefig(semantic_preview_path, dpi=100)
        plt.close()
        print(f"Semantic preview saved to: {semantic_preview_path}")

    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 70)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"Output saved to: {output_dir}")
    print()
    print("Generated files:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in sorted(files):
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size/1024/1024:.2f} MB)")

    print()
    print("=" * 70)
    print("Visualization Instructions")
    print("=" * 70)
    print("To visualize the semantic point cloud:")
    print()
    print("1. CloudCompare (recommended, free):")
    print("   - Download: https://www.cloudcompare.org/")
    print("   - Open: semantic_test_output/semantic_pointcloud.ply")
    print("   - Use: Properties -> Scalar Field to view semantic labels")
    print()
    print("2. MeshLab:")
    print("   - Open the PLY file")
    print("   - Use: Rendering -> Show Face Scalar")
    print()
    print("3. Python:")
    print("   import trimesh")
    print("   pc = trimesh.load('semantic_test_output/semantic_pointcloud.ply')")
    print("   print(pc.vertices.shape)")
    print()


if __name__ == "__main__":
    final_test()
