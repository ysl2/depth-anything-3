#!/usr/bin/env python3
"""
分析当前三维重建结果，找出问题所在
"""
import numpy as np
import sys
sys.path.insert(0, 'src')

def analyze_ply(ply_path):
    """分析PLY文件"""
    print(f"分析 {ply_path}...")

    # 手动解析PLY文件
    with open(ply_path, 'rb') as f:
        # 读取头部
        num_vertices = 0
        has_semantic = False

        for _ in range(50):
            line = f.readline().decode('ascii').strip()
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])
            elif 'property uchar semantic_label' in line:
                has_semantic = True
            elif line == 'end_header':
                break

        print(f"  顶点数: {num_vertices:,}")
        print(f"  有语义标签: {has_semantic}")

        # 读取数据
        dtype = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('red', np.uint8),
            ('green', np.uint8),
            ('blue', np.uint8),
        ])

        if has_semantic:
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
        points = np.stack([data['x'], data['y'], data['z']], axis=1)

        print(f"  实际读取点数: {len(points):,}")

        # 分析点云范围
        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)
        xyz_range = xyz_max - xyz_min

        print(f"\n  点云范围:")
        print(f"    X: [{xyz_min[0]:.2f}, {xyz_max[0]:.2f}] 跨度={xyz_range[0]:.2f}m")
        print(f"    Y: [{xyz_min[1]:.2f}, {xyz_max[1]:.2f}] 跨度={xyz_range[1]:.2f}m")
        print(f"    Z: [{xyz_min[2]:.2f}, {xyz_max[2]:.2f}] 跨度={xyz_range[2]:.2f}m")

        # 分析点密度
        bbox_volume = xyz_range[0] * xyz_range[1] * xyz_range[2]
        density = len(points) / bbox_volume if bbox_volume > 0 else 0
        print(f"\n  点云密度: {density:.2f} 点/立方米")

        # 分析深度分布
        depths = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        print(f"\n  深度分布:")
        print(f"    最小深度: {depths.min():.2f}m")
        print(f"    最大深度: {depths.max():.2f}m")
        print(f"    平均深度: {depths.mean():.2f}m")
        print(f"    中位数深度: {np.median(depths):.2f}m")

        # 检查点的空间分布
        origin_count = np.sum((np.abs(points[:, 0]) < 0.1) &
                             (np.abs(points[:, 1]) < 0.1) &
                             (np.abs(points[:, 2]) < 0.1))
        print(f"\n  原点附近的点数 (±0.1m): {origin_count:,}")

        return points, data if has_semantic else None

def main():
    print("=" * 70)
    print("三维重建结果分析")
    print("=" * 70)
    print()

    ply_path = "./semantic_test_output/semantic_pointcloud.ply"

    if not os.path.exists(ply_path):
        print(f"文件不存在: {ply_path}")
        return

    points, semantic_data = analyze_ply(ply_path)

    print("\n" + "=" * 70)
    print("可能的问题:")
    print("=" * 70)
    print("""
1. 点云覆盖不全 - 可能原因:
   - 相机视角覆盖不足
   - 特征点匹配失败
   - 深度估计在某些区域失败

2. 点云密度不均 - 可能原因:
   - 深度估计质量不稳定
   - 相机位姿优化不收敛
   - 运动恢复结构(SfM)失败

3. 点云有噪点 - 可能原因:
   - 深度估计误差
   - 特征匹配错误
   - 相机标定误差

优化建议:
1. 使用更多输入图像 (当前8张，建议20-50张)
2. 调整DA3的参数 (chunk_size, overlap, ref_view_strategy)
3. 使用use_ray_pose=True获得更准确的位姿
4. 检查图像质量和视角分布
    """)

if __name__ == "__main__":
    import os
    main()
