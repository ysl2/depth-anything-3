# DJI 视频深度估计与可视化完整计划

**时间戳：** 2026-02-07 17:25

## Context

**项目目标：** 从 DJI Mini 3 Pro 航拍视频中提取深度信息，生成 3D 点云并可视化展示。

**当前状态：**
- 9 个 DJI Mini 3 Pro 视频（DJI_0522 ~ DJI_0530），总时长约 43 分钟
- ✅ 抽帧已完成：`/home/songliyu/Templates/DJI-Mini3-Pro/20260204/102MEDIA/fps_2/images/`，共 5,188 帧（2 FPS）
- ✅ 抽帧占用存储：6.1 GB
- 剩余磁盘空间：459 GB

**参考经验：**
- 之前成功处理过 100MEDIA 数据集（143 张图像，512px 降采样）
- 使用 `base_config_16gb.yaml` 配置
- 生成 `combined_pcd.ply` 点云文件后可视化

## Plan

### Step 1: 图像预处理（降采样到 512px）

```python
import os
import glob
from PIL import Image

src_dir = "/home/songliyu/Templates/DJI-Mini3-Pro/20260204/102MEDIA/fps_2/images"
dst_dir = "/home/songliyu/Documents/Depth-Anything-3/data/dji_20260204/images"
target_width = 512

os.makedirs(dst_dir, exist_ok=True)

img_files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")) +
                   glob.glob(os.path.join(src_dir, "*.JPG")))

print(f"Found {len(img_files)} images")

for i, img_path in enumerate(img_files):
    img = Image.open(img_path)
    w, h = img.size
    scale = target_width / w
    new_h = int(h * scale)
    img_resized = img.resize((target_width, new_h), Image.LANCZOS)

    basename = os.path.basename(img_path)
    new_name = os.path.splitext(basename)[0].lower() + ".jpg"
    img_resized.save(os.path.join(dst_dir, new_name))

    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(img_files)} images")

print(f"Done! Resized to {target_width}px width")
```

**输出目录：** `/home/songliyu/Documents/Depth-Anything-3/data/dji_20260204/images/`

### Step 2: 运行 DA3-Streaming 深度估计

使用现有配置 `base_config_16gb.yaml`：

```bash
cd /home/songliyu/Documents/Depth-Anything-3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python da3_streaming/da3_streaming.py \
    --image_dir ./data/dji_20260204/images \
    --config ./da3_streaming/configs/base_config_16gb.yaml \
    --output_dir ./output/dji_20260204
```

**预计输出：**
- `camera_poses.txt` - 相机位姿
- `intrinsic.txt` - 相机内参
- `pcd/combined_pcd.ply` - **合并的彩色点云文件**
- `camera_poses.ply` - 相机位姿点云
- `results_output/frame_*.npz` - 每帧深度、置信度、RGB

### Step 3: 创建 Open3D 可视化脚本

**脚本路径：** `/home/songliyu/Documents/Depth-Anything-3/visualize_o3d.py`

参考 `visualize_pyvista.py` 的功能，使用 Open3D 实现：
- 命令行参数支持
- 点云统计信息输出
- 自动居中显示
- X 轴旋转修正（上下颠倒）
- 可调节点大小、透明度、窗口尺寸

```python
import argparse
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Visualize point cloud using Open3D")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to PLY point cloud file")
    parser.add_argument("--width", type=int, default=1920,
                        help="Window width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Window height (default: 1080)")
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="Point rendering size (default: 2.0)")
    parser.add_argument("--no-rotate", action="store_true",
                        help="Disable X-axis 180° rotation")
    parser.add_argument("--no-center", action="store_true",
                        help="Disable centering")

    args = parser.parse_args()

    # Read point cloud
    print(f"Loading point cloud from {args.input}...")
    pcd = o3d.io.read_point_cloud(args.input)
    print(f"Loaded {len(pcd.points)} points")

    # Get color info
    if pcd.has_colors():
        print("Point cloud has RGB colors")
    else:
        print("Point cloud has no colors, using default coloring")

    # Point cloud bounds
    bounds = pcd.get_axis_aligned_bounding_box()
    min_bound = bounds.min_bound
    max_bound = bounds.max_bound
    print(f"Bounds: X[{min_bound[0]:.2f}, {max_bound[0]:.2f}], "
          f"Y[{min_bound[1]:.2f}, {max_bound[1]:.2f}], "
          f"Z[{min_bound[2]:.2f}, {max_bound[2]:.2f}]")

    # Center the point cloud
    if not args.no_center:
        center = bounds.get_center()
        pcd.translate(-center)
        print(f"Centered point cloud at [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

    # Fix upside-down: rotate 180° around X axis
    if not args.no_rotate:
        R = pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
        pcd.rotate(R, center=[0, 0, 0])
        print("Applied X-axis 180° rotation")

    # Set point size using render option
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Point Cloud Viewer",
                      width=args.width, height=args.height)
    vis.add_geometry(pcd)

    # Get render options and set point size
    opt = vis.get_render_option()
    opt.point_size = args.point_size

    print("Starting visualization...")
    print("Controls:")
    print("  - Mouse drag: Rotate view")
    print("  - Mouse wheel: Zoom")
    print("  - Shift + drag: Pan")
    print("  - Q / ESC: Exit")

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
```

### Step 4: 运行可视化

```bash
cd /home/songliyu/Documents/Depth-Anything-3
python visualize_o3d.py -i ./output/dji_20260204/pcd/combined_pcd.ply
```

## Critical Files

| 文件 | 说明 |
|------|------|
| `da3_streaming/configs/base_config_16gb.yaml` | 使用现有配置 |
| `visualize_pyvista.py` | 参考可视化脚本 |
| `visualize_o3d.py` | 新建：Open3D 可视化脚本 |
| `predict1.md` | 100MEDIA 处理记录参考 |

## Expected Output

```
./output/dji_20260204/
├── camera_poses.txt          # 相机位姿
├── intrinsic.txt             # 相机内参
├── camera_poses.ply          # 相机位姿点云
├── pcd/
│   ├── combined_pcd.ply      # 合并点云（主要可视化目标）
│   ├── 0_pcd.ply            # 各 chunk 点云
│   └── ...
└── results_output/
    └── frame_*.npz          # 每帧结果
```

## Verification

1. **点云生成验证：**
   ```bash
   ls -lh ./output/dji_20260204/pcd/combined_pcd.ply
   ```

2. **点云质量检查：**
   ```bash
   python -c "import open3d as o3d; pcd=o3d.io.read_point_cloud('./output/dji_20260204/pcd/combined_pcd.ply'); print(f'Points: {len(pcd.points)}, Colors: {pcd.has_colors()}')"
   ```

3. **可视化验证：**
   运行 `visualize_o3d.py`，检查点云正常加载和显示

## 备选可视化方案

如果 Open3D 无法正常显示 GUI，可使用：
- **PyVista**：`python visualize_pyvista.py -i ./output/dji_20260204/pcd/combined_pcd.ply`
- **MeshLab**：`meshlab ./output/dji_20260204/pcd/combined_pcd.ply`

## 存储预估

| 阶段 | 存储占用 |
|------|----------|
| 原始 2fps 帧 | 6.1 GB（已完成） |
| 降采样图像 | ~1 GB |
| DA3-Streaming 临时文件 | ~30-50 GB（自动清理） |
| 最终输出 | ~2-5 GB |
| **总计** | ~40 GB（剩余 459 GB 充足） |
