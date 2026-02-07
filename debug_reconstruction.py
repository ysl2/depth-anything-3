#!/usr/bin/env python3
"""
检查改进的重建数据
"""
import numpy as np
import sys
sys.path.insert(0, 'src')

# 从improved_reconstruction.py中提取关键逻辑来检查
import os
import sys
import gc
import torch
from PIL import Image

sys.path.insert(0, 'src')

from depth_anything_3.api import DepthAnything3

def debug_reconstruction():
    """调试重建问题"""

    print("=" * 70)
    print("调试三维重建")
    print("=" * 70)

    # 配置 - 先用少量图像调试
    image_dir = "/home/songliyu/Templates/south-building/images"
    num_images = 16  # 先用16张调试

    # 加载图像
    print(f"\n[1/3] 加载图像...")
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

    # 运行DA3
    print(f"\n[2/3] 运行DA3...")

    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    model = model.to("cuda")

    with torch.no_grad():
        prediction = model.inference(
            processed_images,
            ref_view_strategy="saddle_balanced",
            use_ray_pose=True,
        )

    print(f"  深度图形状: {prediction.depth.shape}")
    print(f"  内参形状: {prediction.intrinsics.shape}")
    print(f"  外参形状: {prediction.extrinsics.shape}")

    # 分析第一张图像的深度
    depth0 = prediction.depth[0]
    print(f"\n  图像1深度统计:")
    print(f"    最小值: {depth0.min():.4f}")
    print(f"    最大值: {depth0.max():.4f}")
    print(f"    平均值: {depth0.mean():.4f}")
    print(f"    中位数: {np.median(depth0):.4f}")

    # 检查外参
    ext0 = prediction.extrinsics[0]
    print(f"\n  图像1外参 (c2w):")
    print(f"    {ext0}")

    # 检查内参
    int0 = prediction.intrinsics[0]
    print(f"\n  图像1内参:")
    print(f"    fx={int0[0,0]:.2f}, fy={int0[1,1]:.2f}")
    print(f"    cx={int0[0,2]:.2f}, cy={int0[1,2]:.2f}")

    # 生成点云
    print(f"\n[3/3] 生成点云...")

    all_points = []

    for img_idx in range(len(processed_images)):
        depth = prediction.depth[img_idx]
        intrinsics = prediction.intrinsics[img_idx]
        extrinsics = prediction.extrinsics[img_idx]

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

        print(f"  图像{img_idx+1}: 总点={H*W}, 有效点={valid.sum()}, "
              f"深度范围=[{depth.min():.2f}, {depth.max():.2f}], "
              f"点云范围=[{points[valid, 0].min():.2f}, {points[valid, 0].max():.2f}] x "
              f"[{points[valid, 1].min():.2f}, {points[valid, 1].max():.2f}] x "
              f"[{points[valid, 2].min():.2f}, {points[valid, 2].max():.2f}]")

        all_points.append(points[valid])

    # 合并所有点
    all_points = np.concatenate(all_points, axis=0)

    print(f"\n  总点数: {len(all_points):,}")

    # 分析点云范围
    xyz_min = all_points.min(axis=0)
    xyz_max = all_points.max(axis=0)
    xyz_range = xyz_max - xyz_min

    print(f"\n  点云范围:")
    print(f"    X: [{xyz_min[0]:.2f}, {xyz_max[0]:.2f}] 跨度={xyz_range[0]:.2f}m")
    print(f"    Y: [{xyz_min[1]:.2f}, {xyz_max[1]:.2f}] 跨度={xyz_range[1]:.2f}m")
    print(f"    Z: [{xyz_min[2]:.2f}, {xyz_max[2]:.2f}] 跨度={xyz_range[2]:.2f}m")

    # 检查点的分布
    origin_count = np.sum((np.abs(all_points[:, 0]) < 0.5) &
                         (np.abs(all_points[:, 1]) < 0.5) &
                         (np.abs(all_points[:, 2]) < 0.5))
    print(f"\n  原点附近的点数 (±0.5m): {origin_count:,}")

    # 清理
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return all_points

if __name__ == "__main__":
    points = debug_reconstruction()
