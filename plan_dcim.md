# 在 DCIM 数据集上执行 DA3-Streaming 流式预测
## 执行计划

---

### 一、数据集与环境信息

| 项目 | 详情 |
|------|------|
| **数据集** | `./data/DCIM/images/` |
| **图像数量** | 825张 |
| **原始分辨率** | 4032 x 3024（需降采样） |
| **文件扩展名** | `.JPG`（大写，需转小写） |
| **GPU** | NVIDIA RTX 5060 Ti 16GB (16311 MiB) |
| **工作目录** | `/home/songliyu/Documents/Depth-Anything-3` |
| **输出目录** | `./output/dcim` |

---

### 二、执行步骤

#### 步骤 1：图像预处理（降采样 + 扩展名转换）

**原因**：
1. 原始分辨率 4032x3024 过高，会导致显存溢出
2. 代码只支持小写 `.jpg`/`.png` 扩展名
3. 825张图像需要高效处理

**预处理方案**：
```bash
mkdir -p data/DCIM/images_resized

python3 << 'EOF'
import os
import glob
from PIL import Image

src_dir = "data/DCIM/images"
dst_dir = "data/DCIM/images_resized"
target_width = 640  # 适中的分辨率，平衡质量和速度

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

    if (i+1) % 100 == 0:
        print(f"Processed {i+1}/{len(img_files)} images")

print(f"Done! Resized to {target_width}px width, saved as lowercase .jpg")
EOF
```

---

#### 步骤 2：创建配置文件

针对825张大图像集的优化配置：

```bash
cat > da3_streaming/configs/dcim.yaml << 'EOF'
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'

Model:
  chunk_size: 30          # 适中的chunk大小
  overlap: 15             # chunk间重叠
  loop_chunk_size: 30     # 回环chunk大小
  loop_enable: False      # 禁用回环检测以节省显存
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
    sample_ratio: 0.01          # 降低采样率以控制点云大小
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
EOF
```

**配置说明**：
- `chunk_size: 30` - 平衡显存和效率
- `loop_enable: False` - 禁用回环以节省显存（大图像集不需要回环也能获得好结果）
- `sample_ratio: 0.01` - 降低采样率控制点云大小

---

#### 步骤 3：执行流式预测

```bash
cd /home/songliyu/Documents/Depth-Anything-3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python da3_streaming/da3_streaming.py \
    --image_dir ./data/DCIM/images_resized \
    --config ./da3_streaming/configs/dcim.yaml \
    --output_dir ./output/dcim
```

---

#### 步骤 4：可视化结果

创建 `visualize_dcim.py` 可视化脚本：

```python
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 读取点云
pcd = o3d.io.read_point_cloud("./output/dcim/pcd/combined_pcd.ply")
print(f"Loaded {len(pcd.points)} points")

# 获取点云边界框进行居中显示
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
pcd.translate(-center)

# 可视化
o3d.visualization.draw_geometries(
    [pcd],
    window_name="DCIM 3D Reconstruction",
    width=1920,
    height=1080
)
```

---

### 三、预期输出

运行完成后，`./output/dcim/` 目录将包含：

| 文件 | 说明 |
|------|------|
| `camera_poses.txt` | 相机位姿 |
| `intrinsic.txt` | 相机内参 |
| `pcd/combined_pcd.ply` | 合并的彩色点云文件 |
| `results_output/frame_*.npz` | 每帧的深度图、置信度、RGB图像 |

---

### 四、资源消耗预估

| 项目 | 预估值 |
|------|--------|
| **处理时间** | 2-5 分钟（825帧） |
| **显存峰值** | 12-15 GB |
| **磁盘占用** | 5-10 GB |

---

### 五、配置优化说明

针对825张图像的优化：

1. **禁用回环检测** (`loop_enable: False`)
   - 回环检测需要额外显存加载约2×chunk_size的图像
   - 对于线性扫描的图像序列，回环收益有限

2. **适中的chunk_size** (30)
   - 太小：处理速度慢，对齐效果差
   - 太大：显存可能溢出
   - 30是16GB显存的安全值

3. **降低采样率** (`sample_ratio: 0.01`)
   - 825张图像的点云会很大
   - 1%采样率平衡可视化和文件大小
