# DCIM 数据集 DA3-Streaming 流式预测执行记录

## 环境信息

| 项目 | 详情 |
|------|------|
| **数据集** | `./data/DCIM/images/` |
| **图像数量** | 825张 |
| **原始分辨率** | 4032 x 3024 |
| **预处理分辨率** | 640 x 480 |
| **文件扩展名** | `.JPG` → `.jpg` |
| **GPU** | NVIDIA RTX 5060 Ti 16GB |
| **工作目录** | `/home/songliyu/Documents/Depth-Anything-3` |
| **输出目录** | `./output/dcim` |

---

## 执行步骤

### 步骤 1：图像预处理

```bash
mkdir -p data/DCIM/images_resized

python3 << 'EOF'
import os
import glob
from PIL import Image

src_dir = "data/DCIM/images"
dst_dir = "data/DCIM/images_resized"
target_width = 640

os.makedirs(dst_dir, exist_ok=True)

img_files = sorted(glob.glob(os.path.join(src_dir, "*.JPG")) +
                   glob.glob(os.path.join(src_dir, "*.jpg")) +
                   glob.glob(os.path.join(src_dir, "*.PNG")) +
                   glob.glob(os.path.join(src_dir, "*.png")))

for i, img_path in enumerate(img_files):
    img = Image.open(img_path)
    w, h = img.size
    scale = target_width / w
    new_h = int(h * scale)
    img_resized = img.resize((target_width, new_h), Image.LANCZOS)

    basename = os.path.basename(img_path)
    new_name = os.path.splitext(basename)[0].lower() + ".jpg"
    img_resized.save(os.path.join(dst_dir, new_name), quality=95)

    if (i+1) % 100 == 0:
        print(f"Processed {i+1}/{len(img_files)} images")
EOF
```

**结果**：
- 处理了 825 张图像
- 从 4032x3024 降采样到 640x480
- 转换为大写 .JPG 到小写 .jpg

---

### 步骤 2：创建配置文件

创建 `da3_streaming/configs/dcim.yaml`：

```yaml
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'

Model:
  chunk_size: 30
  overlap: 15
  loop_chunk_size: 30
  loop_enable: False      # 禁用回环检测
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
    sample_ratio: 0.01
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

---

### 步骤 3：执行流式预测

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python da3_streaming/da3_streaming.py \
    --image_dir ./data/DCIM/images_resized \
    --config ./da3_streaming/configs/dcim.yaml \
    --output_dir ./output/dcim
```

**处理过程**：
- 825 张图像分为 54 个 chunk
- 每个 chunk 30 张图像，重叠 15 张
- 使用 Sim3 对齐算法进行 chunk 间配准

**执行时间**: 约 8 分钟

---

## 输出结果

运行完成后，`./output/dcim/` 目录包含：

| 文件 | 说明 |
|------|------|
| `camera_poses.txt` | 相机位姿 (825行) |
| `intrinsic.txt` | 相机内参 |
| `pcd/combined_pcd.ply` | **合并的彩色点云文件** |
| `pcd/0_pcd.ply ~ 53_pcd.ply` | 各chunk的点云文件 (54个) |
| `camera_poses.ply` | 相机位姿点云 |
| `results_output/frame_*.npz` | 每帧的深度图、置信度、RGB (825个) |

---

## 3D 重建结果

### 点云统计

| 指标 | 数值 |
|------|------|
| **点云点数** | 2,220,558 点 |
| **颜色信息** | 有 (RGB) |
| **边界框中心** | [42.94, -1.26, 44.63] |
| **场景尺寸** | 124.16m × 23.46m × 103.59m |

### 可视化

生成多个视角的可视化截图：
- `visualization.png` - 主视角
- `view_1_front.png` - 正视图
- `view_2_top.png` - 俯视图
- `view_3_side.png` - 侧视图
- `view_4_angle.png` - 斜视图

---

## 配置优化说明

针对825张图像的优化：

1. **禁用回环检测** (`loop_enable: False`)
   - 回环检测需要额外显存
   - 对于无人机航拍图像序列，回环收益有限

2. **适中的chunk_size** (30)
   - 平衡显存利用率和处理效率
   - 16GB显存的安全配置

3. **降低采样率** (`sample_ratio: 0.01`)
   - 控制点云文件大小
   - 1%采样率仍能保留足够的细节

---

## 执行总结

| 步骤 | 操作 | 状态 |
|------|------|------|
| 1 | 创建执行计划 | ✅ |
| 2 | 图像预处理 | ✅ |
| 3 | 创建配置文件 | ✅ |
| 4 | 执行流式预测 | ✅ |
| 5 | 可视化结果 | ✅ |
| 6 | 生成执行日志 | ✅ |

**最终配置**：
- chunk_size: 30
- loop_enable: False
- 预处理分辨率: 640px
- 点云数量: 2,220,558 个点
