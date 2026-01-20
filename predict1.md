# 100MEDIA 数据集 DA3-Streaming 流式预测执行记录

## 环境信息

| 项目 | 详情 |
|------|------|
| **数据集** | `data/100MEDIA/` |
| **图像数量** | 143张 |
| **原始分辨率** | 3968 x 2976 |
| **文件扩展名** | `.JPG`（大写） |
| **GPU** | NVIDIA RTX 5060 Ti 16GB (16311 MiB) |
| **工作目录** | `/home/songliyu/Documents/Depth-Anything-3` |
| **输出目录** | `./output/100media` |

---

## 执行步骤

### 步骤 1：数据集检查

数据集已存在于 `./data/100MEDIA/images/`，包含143张 .JPG 格式图像。

```bash
ls ./data/100MEDIA/images/*.JPG | wc -l
# 输出: 143
```

---

### 步骤 2：图像预处理（降采样 + 扩展名转换）

**原因**：
1. 原始分辨率 3968x2976 过高，会导致显存溢出
2. 代码只支持小写 `.jpg`/`.png` 扩展名

```bash
cd /home/songliyu/Documents/Depth-Anything-3
mkdir -p data/100MEDIA/images_resized

python3 << 'EOF'
import os
import glob
from PIL import Image

src_dir = "data/100MEDIA/images"
dst_dir = "data/100MEDIA/images_resized"
target_width = 512  # 降采样目标宽度

os.makedirs(dst_dir, exist_ok=True)

# 获取所有图像文件（支持大小写扩展名）
img_files = sorted(glob.glob(os.path.join(src_dir, "*.JPG")) +
                   glob.glob(os.path.join(src_dir, "*.jpg")) +
                   glob.glob(os.path.join(src_dir, "*.PNG")) +
                   glob.glob(os.path.join(src_dir, "*.png")))

print(f"Found {len(img_files)} images")

for i, img_path in enumerate(img_files):
    img = Image.open(img_path)
    w, h = img.size
    scale = target_width / w
    new_h = int(h * scale)
    img_resized = img.resize((target_width, new_h), Image.LANCZOS)

    # 转换为小写 .jpg 扩展名
    basename = os.path.basename(img_path)
    new_name = os.path.splitext(basename)[0].lower() + ".jpg"
    img_resized.save(os.path.join(dst_dir, new_name))

    if (i+1) % 20 == 0:
        print(f"Processed {i+1}/{len(img_files)} images")

print(f"Done! Resized to {target_width}px width, saved as lowercase .jpg")
EOF
```

**输出**：
```
Found 143 images
Processed 20/143 images
Processed 40/143 images
...
Done! Resized to 512px width, saved as lowercase .jpg
```

---

### 步骤 3：创建配置文件

创建 `da3_streaming/configs/100media.yaml`（基于之前 south-building 的成功经验）：

```yaml
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'

Model:
  chunk_size: 20
  overlap: 10
  loop_chunk_size: 20
  loop_enable: False
  useDBoW: False
  delete_temp_files: True
  align_lib: 'triton'
  align_method: 'sim3'
  scale_compute_method: 'auto'
  align_type: 'dense'

  ref_view_strategy: 'saddle_balanced'
  ref_view_strategy_loop: 'saddle_balanced'
  depth_threshold: 15.0

  save_depth_conf_result: True
  save_debug_info: False

  Sparse_Align:
    keypoint_select: 'orb'
    keypoint_num: 5000

  IRLS:
    delta: 0.1
    max_iters: 5
    tol: 1e-9

  Pointcloud_Save:
    sample_ratio: 0.015
    conf_threshold_coef: 0.75

Loop:
  SALAD:
    image_size: [336, 336]
    batch_size: 32
    similarity_threshold: 0.85
    top_k: 5
    use_nms: True
    nms_threshold: 25

  SIM3_Optimizer:
    lang_version: 'cpp'
    max_iterations: 30
    lambda_init: 1e-6
```

**配置说明**：
- `chunk_size: 20` - 基于 south-building 数据集的显存测试结果
- `loop_enable: False` - 禁用回环检测以避免显存溢出
- 其他参数保持默认值

---

### 步骤 4：执行流式预测

```bash
cd /home/songliyu/Documents/Depth-Anything-3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python da3_streaming/da3_streaming.py \
    --image_dir ./data/100MEDIA/images_resized \
    --config ./da3_streaming/configs/100media.yaml \
    --output_dir ./output/100media
```

**处理信息**：
- 图像数量：143张
- 分块数量：14个chunks（每块20张，重叠10张）
- 处理时间：约60秒

---

## 输出结果

运行完成后，`./output/100media/` 目录包含：

| 文件 | 说明 |
|------|------|
| `camera_poses.txt` | 相机位姿（每行12个参数，3x4外参矩阵，w2c格式） |
| `intrinsic.txt` | 相机内参（每行 fx, fy, cx, cy） |
| `pcd/combined_pcd.ply` | **合并的彩色点云文件** (516,265 个点) |
| `pcd/0_pcd.ply, 1_pcd.ply, ...` | 各chunk的点云文件 |
| `camera_poses.ply` | 相机位姿点云 |
| `results_output/frame_*.npz` | 每帧的深度图、置信度、RGB图像、内参 |

---

## 可视化重建结果

### 问题1：点云上下颠倒

**解决方案**：绕 X 轴旋转 180 度修正方向

### 可视化代码

```python
import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud("./output/100media/pcd/combined_pcd.ply")
print(f"Loaded {len(pcd.points)} points")

# 获取点云边界框进行居中显示
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
pcd.translate(-center)

# 修正上下颠倒：绕X轴旋转180度
R = pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
pcd.rotate(R, center=[0, 0, 0])

# 可视化
o3d.visualization.draw_geometries(
    [pcd],
    window_name="100MEDIA 3D Reconstruction (Corrected)",
    width=1920,
    height=1080
)
```

### 操作方式

| 操作 | 按键/鼠标 |
|------|-----------|
| 旋转视角 | 鼠标左键拖拽 |
| 缩放 | 鼠标滚轮 |
| 平移 | Shift + 鼠标拖拽 |
| 更多帮助 | 按 `H` |

---

## 与 South-Building 数据集对比

| 项目 | South-Building | 100MEDIA |
|------|----------------|----------|
| 图像数量 | 128张 | 143张 |
| 原始分辨率 | 3072 x 2304 | 3968 x 2976 |
| 点云数量 | 397,588 | 516,265 |
| chunk数量 | 12 | 14 |
| 显存配置 | chunk_size=20, loop=False | chunk_size=20, loop=False |

---

## 执行总结

| 步骤 | 操作 | 状态 |
|------|------|------|
| 1 | 检查数据集 | ✅ |
| 2 | 图像预处理 | ✅ |
| 3 | 创建配置文件 | ✅ |
| 4 | 执行流式预测 | ✅ |
| 5 | 可视化重建结果 | ✅ |

**最终配置**：
- chunk_size: 20
- loop_enable: False
- 点云数量: 516,265 个点
- 数据集: 100MEDIA (143张图像)

---

## 经验总结

1. **配置复用**：基于之前 south-building 数据集的成功配置直接复用
2. **显存优化**：chunk_size=20, loop_enable=False 是16GB显存的安全配置
3. **预处理**：必须降采样到512px宽度并转换为小写扩展名
4. **可视化**：需要绕X轴旋转180度修正上下颠倒问题
