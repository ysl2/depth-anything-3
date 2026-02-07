# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
语义点云保存模块

扩展PLY格式支持语义标签，支持保存带有语义信息的点云。
"""

import os
import numpy as np
from typing import Optional, Tuple


def write_ply_header_with_semantics(f, num_vertices, has_semantics=True):
    """
    写入带语义标签的PLY文件头

    Args:
        f: 文件对象
        num_vertices: 顶点数量
        has_semantics: 是否包含语义标签
    """
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]

    if has_semantics:
        header.append("property uchar semantic_label")
        header.append("property uchar semantic_red")
        header.append("property uchar semantic_green")
        header.append("property uchar semantic_blue")

    header.append("end_header")
    f.write("\n".join(header).encode() + b"\n")


def write_ply_batch_with_semantics(f, points, colors, semantics=None, semantic_colors=None):
    """
    写入带语义标签的点云批次

    Args:
        f: 文件对象
        points: (N, 3) 点坐标
        colors: (N, 3) RGB颜色
        semantics: (N,) 语义标签 (可选)
        semantic_colors: (N, 3) 语义颜色 (可选)
    """
    if semantics is None or semantic_colors is None:
        # 不带语义标签，使用原始格式
        dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
        ]
        structured = np.zeros(len(points), dtype=dtype)
        structured["x"] = points[:, 0]
        structured["y"] = points[:, 1]
        structured["z"] = points[:, 2]
        structured["red"] = colors[:, 0]
        structured["green"] = colors[:, 1]
        structured["blue"] = colors[:, 2]
    else:
        # 带语义标签
        dtype = [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
            ("semantic_label", np.uint8),
            ("semantic_red", np.uint8),
            ("semantic_green", np.uint8),
            ("semantic_blue", np.uint8),
        ]
        structured = np.zeros(len(points), dtype=dtype)
        structured["x"] = points[:, 0]
        structured["y"] = points[:, 1]
        structured["z"] = points[:, 2]
        structured["red"] = colors[:, 0]
        structured["green"] = colors[:, 1]
        structured["blue"] = colors[:, 2]
        structured["semantic_label"] = semantics
        structured["semantic_red"] = semantic_colors[:, 0]
        structured["semantic_green"] = semantic_colors[:, 1]
        structured["semantic_blue"] = semantic_colors[:, 2]

    f.write(structured.tobytes())


def save_confident_pointcloud_batch_with_semantics(
    points: np.ndarray,
    colors: np.ndarray,
    confs: np.ndarray,
    semantics: Optional[np.ndarray],
    semantic_colors: Optional[np.ndarray],
    output_path: str,
    conf_threshold: float,
    sample_ratio: float = 1.0,
    batch_size: int = 1000000,
    colormap: Optional[np.ndarray] = None,
):
    """
    保存带语义标签的置信点云

    Args:
        points: (b, H, W, 3) 或 (N, 3) 点坐标
        colors: (b, H, W, 3) 或 (N, 3) RGB颜色
        confs: (b, H, W) 或 (N,) 置信度
        semantics: (b, H, W) 或 (N,) 语义标签 (可选)
        semantic_colors: (b, H, W, 3) 或 (N, 3) 语义颜色 (可选)
        output_path: 输出路径
        conf_threshold: 置信度阈值
        sample_ratio: 采样比例
        batch_size: 批次大小
        colormap: 语义颜色映射 (num_classes, 3)
    """
    # 处理输入维度
    if points.ndim == 2:
        b = 1
        points = points[np.newaxis, ...]
        colors = colors[np.newaxis, ...]
        confs = confs[np.newaxis, ...]
        if semantics is not None:
            semantics = semantics[np.newaxis, ...]
        if semantic_colors is not None:
            semantic_colors = semantic_colors[np.newaxis, ...]
    elif points.ndim == 4:
        b = points.shape[0]
    else:
        raise ValueError("Unsupported points dimension. Must be 2 (N,3) or 4 (b,H,W,3)")

    # 统计有效点数
    total_valid = 0
    for i in range(b):
        cfs = confs[i].reshape(-1)
        total_valid += np.count_nonzero((cfs >= conf_threshold) & (cfs > 1e-5))

    # 计算采样数
    if sample_ratio < 1.0:
        num_samples = int(total_valid * sample_ratio)
    else:
        num_samples = total_valid

    if num_samples == 0:
        save_ply_with_semantics(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            output_path
        )
        return

    has_semantics = (semantics is not None) and (semantic_colors is not None)

    # 打开文件写入
    with open(output_path, "wb") as f:
        write_ply_header_with_semantics(f, num_samples, has_semantics=has_semantics)

        if sample_ratio == 1.0:
            # 不采样，直接写入
            for i in range(b):
                pts = points[i].reshape(-1, 3).astype(np.float32)
                cls = colors[i].reshape(-1, 3).astype(np.uint8)
                cfs = confs[i].reshape(-1).astype(np.float32)

                # 获取语义信息
                if has_semantics:
                    sems = semantics[i].reshape(-1).astype(np.uint8)
                    sem_cls = semantic_colors[i].reshape(-1, 3).astype(np.uint8)
                else:
                    sems = None
                    sem_cls = None

                # 过滤有效点
                mask = (cfs >= conf_threshold) & (cfs > 1e-5)
                valid_pts = pts[mask]
                valid_cls = cls[mask]
                valid_sems = sems[mask] if sems is not None else None
                valid_sem_cls = sem_cls[mask] if sem_cls is not None else None

                # 批次写入
                for j in range(0, len(valid_pts), batch_size):
                    batch_pts = valid_pts[j:j+batch_size]
                    batch_cls = valid_cls[j:j+batch_size]
                    batch_sems = valid_sems[j:j+batch_size] if valid_sems is not None else None
                    batch_sem_cls = valid_sem_cls[j:j+batch_size] if valid_sem_cls is not None else None

                    write_ply_batch_with_semantics(f, batch_pts, batch_cls, batch_sems, batch_sem_cls)
        else:
            # 水塘采样
            reservoir_pts = np.zeros((num_samples, 3), dtype=np.float32)
            reservoir_clr = np.zeros((num_samples, 3), dtype=np.uint8)
            if has_semantics:
                reservoir_sems = np.zeros((num_samples,), dtype=np.uint8)
                reservoir_sem_clr = np.zeros((num_samples, 3), dtype=np.uint8)
            else:
                reservoir_sems = None
                reservoir_sem_clr = None

            count = 0

            for i in range(b):
                pts = points[i].reshape(-1, 3).astype(np.float32)
                cls = colors[i].reshape(-1, 3).astype(np.uint8)
                cfs = confs[i].reshape(-1).astype(np.float32)

                if has_semantics:
                    sems = semantics[i].reshape(-1).astype(np.uint8)
                    sem_cls = semantic_colors[i].reshape(-1, 3).astype(np.uint8)
                else:
                    sems = None
                    sem_cls = None

                mask = (cfs >= conf_threshold) & (cfs > 1e-5)
                valid_pts = pts[mask]
                valid_cls = cls[mask]
                valid_sems = sems[mask] if sems is not None else None
                valid_sem_cls = sem_cls[mask] if sem_cls is not None else None

                n_valid = len(valid_pts)

                if count < num_samples:
                    fill_count = min(num_samples - count, n_valid)

                    reservoir_pts[count:count + fill_count] = valid_pts[:fill_count]
                    reservoir_clr[count:count + fill_count] = valid_cls[:fill_count]
                    if has_semantics:
                        reservoir_sems[count:count + fill_count] = valid_sems[:fill_count]
                        reservoir_sem_clr[count:count + fill_count] = valid_sem_cls[:fill_count]
                    count += fill_count

                    if fill_count < n_valid:
                        remaining_pts = valid_pts[fill_count:]
                        remaining_cls = valid_cls[fill_count:]
                        remaining_sems = valid_sems[fill_count:] if valid_sems is not None else None
                        remaining_sem_cls = valid_sem_cls[fill_count:] if valid_sem_cls is not None else None

                        count, reservoir_pts, reservoir_clr, reservoir_sems, reservoir_sem_clr = \
                            optimized_vectorized_reservoir_sampling_with_semantics(
                                remaining_pts, remaining_cls, remaining_sems, remaining_sem_cls,
                                count, reservoir_pts, reservoir_clr, reservoir_sems, reservoir_sem_clr
                            )
                else:
                    count, reservoir_pts, reservoir_clr, reservoir_sems, reservoir_sem_clr = \
                        optimized_vectorized_reservoir_sampling_with_semantics(
                            valid_pts, valid_cls, valid_sems, valid_sem_cls,
                            count, reservoir_pts, reservoir_clr, reservoir_sems, reservoir_sem_clr
                        )

            save_ply_with_semantics(
                reservoir_pts, reservoir_clr, output_path,
                semantics=reservoir_sems, semantic_colors=reservoir_sem_clr
            )

    print(f"Saved semantic point cloud with {num_samples} points to {output_path}")


def optimized_vectorized_reservoir_sampling_with_semantics(
    new_points: np.ndarray,
    new_colors: np.ndarray,
    new_semantics: Optional[np.ndarray],
    new_semantic_colors: Optional[np.ndarray],
    current_count: int,
    reservoir_points: np.ndarray,
    reservoir_colors: np.ndarray,
    reservoir_semantics: Optional[np.ndarray] = None,
    reservoir_semantic_colors: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    带语义标签的水塘采样

    Args:
        new_points: 新点坐标 (M, 3)
        new_colors: 新RGB颜色 (M, 3)
        new_semantics: 新语义标签 (M,) (可选)
        new_semantic_colors: 新语义颜色 (M, 3) (可选)
        current_count: 当前计数
        reservoir_points: 水塘点 (K, 3)
        reservoir_colors: 水塘RGB颜色 (K, 3)
        reservoir_semantics: 水塘语义标签 (K,) (可选)
        reservoir_semantic_colors: 水塘语义颜色 (K, 3) (可选)

    Returns:
        (更新计数, 更新水塘点, 更新水塘颜色, 更新水塘语义, 更新水塘语义颜色)
    """
    random_gen = np.random
    reservoir_size = len(reservoir_points)
    num_new_points = len(new_points)

    if num_new_points == 0:
        return current_count, reservoir_points, reservoir_colors, reservoir_semantics, reservoir_semantic_colors

    # 计算每个点的索引
    point_indices = np.arange(current_count + 1, current_count + num_new_points + 1)

    # 生成随机数
    random_values = random_gen.randint(0, point_indices, size=num_new_points)

    # 确定替换位置
    replacement_mask = random_values < reservoir_size
    replacement_positions = random_values[replacement_mask]

    # 应用替换
    if np.any(replacement_mask):
        points_to_replace = new_points[replacement_mask]
        colors_to_replace = new_colors[replacement_mask]

        reservoir_points[replacement_positions] = points_to_replace
        reservoir_colors[replacement_positions] = colors_to_replace

        if new_semantics is not None and reservoir_semantics is not None:
            semantics_to_replace = new_semantics[replacement_mask]
            sem_colors_to_replace = new_semantic_colors[replacement_mask]
            reservoir_semantics[replacement_positions] = semantics_to_replace
            reservoir_semantic_colors[replacement_positions] = sem_colors_to_replace

    return (
        current_count + num_new_points,
        reservoir_points,
        reservoir_colors,
        reservoir_semantics,
        reservoir_semantic_colors
    )


def save_ply_with_semantics(
    points: np.ndarray,
    colors: np.ndarray,
    filename: str,
    semantics: Optional[np.ndarray] = None,
    semantic_colors: Optional[np.ndarray] = None,
):
    """
    保存PLY文件（支持语义标签）

    Args:
        points: (N, 3) 点坐标
        colors: (N, 3) RGB颜色
        filename: 输出文件名
        semantics: (N,) 语义标签 (可选)
        semantic_colors: (N, 3) 语义颜色 (可选)
    """
    with open(filename, "wb") as f:
        write_ply_header_with_semantics(f, len(points), has_semantics=(semantics is not None))
        write_ply_batch_with_semantics(f, points, colors, semantics, semantic_colors)


def merge_semantic_ply_files(input_dir: str, output_path: str):
    """
    合并带语义标签的PLY文件

    Args:
        input_dir: 输入目录
        output_path: 输出文件路径
    """
    import glob

    print("Merging semantic PLY files...")

    input_files = sorted(glob.glob(os.path.join(input_dir, "*_pcd.ply")))

    if not input_files:
        print("No PLY files found")
        return

    # 统计顶点数
    total_vertices = 0
    has_semantics = False

    for file in input_files:
        with open(file, "rb") as f:
            for line in f:
                if line.startswith(b"element vertex"):
                    vertex_count = int(line.split()[-1])
                    total_vertices += vertex_count
                elif line.startswith(b"property semantic_label"):
                    has_semantics = True
                elif line.startswith(b"end_header"):
                    break

    # 写入合并文件
    with open(output_path, "wb") as out_f:
        write_ply_header_with_semantics(out_f, total_vertices, has_semantics=has_semantics)

        for file in input_files:
            print(f"Processing {file}")
            with open(file, "rb") as in_f:
                # 跳过头
                in_header = True
                while in_header:
                    line = in_f.readline()
                    if line.startswith(b"end_header"):
                        in_header = False
                data = in_f.read()
                out_f.write(data)

    print(f"Merge completed! Total points: {total_vertices}")
    print(f"Output file: {output_path}")
