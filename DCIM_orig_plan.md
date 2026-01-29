# DJI Mini3 Pro 数据集 (DCIM) DA3-Streaming 流式预测计划

## 最终版计划

---

### 一、数据集与环境信息

| 项目 | 详情 |
|------|------|
| **数据集** | `./data/DCIM/images/` (DJI Mini3 Pro 航拍图像) |
| **图像数量** | 969张 |
| **原始分辨率** | 4032 x 3024 |
| **文件扩展名** | `.JPG`（大写，需要转换） |
| **GPU** | NVIDIA RTX 5060 Ti 16GB (16311 MiB, 空闲 ~15457 MB) |
| **工作目录** | `/home/songliyu/Documents/Depth-Anything-3` |
| **输出目录** | `./output/dcim` |

---

### 二、GPU优化策略

根据参考经验（south-building数据集测试）：

| chunk_size | loop_enabled | 峰值显存 | 结果 |
|------------|--------------|----------|------|
| 30 | True | ~15777 MB | ❌ OOM |
| 20 | True | ~15743 MB | ❌ OOM (回环阶段) |
| 20 | False | ~13703 MB | ✅ 成功 |

**DCIM优化配置**：
- `chunk_size: 25`: 在安全范围内最大化利用显存
- `loop_enable: False`: 避免回环阶段显存翻倍导致OOM
- `target_width: 640`: 平衡质量与显存消耗
- 预计峰值显存: ~14 GB

---

### 三、执行步骤

#### 步骤 1：图像预处理（降采样 + 扩展名转换）

```bash
cd /home/songliyu/Documents/Depth-Anything-3
mkdir -p data/DCIM/images_resized

python3 << 'EOF'
import os
import glob
from PIL import Image

src_dir = "data/DCIM/images"
dst_dir = "data/DCIM/images_resized"
target_width = 640  # 降采样目标宽度

os.makedirs(dst_dir, exist_ok=True)

# 获取所有图像文件（支持大小写扩展名）
img_files = sorted(glob.glob(os.path.join(src_dir, "*.JPG")) +
                   glob.glob(os.path.join(src_dir, "*.jpg")) +
                   glob.glob(os.path.join(src_dir, "*.PNG")) +
                   glob.glob(os.path.join(src_dir, "*.png")))

print(f"Found {len(img_files)} images")

for i, img_path in enumerate(img_files):
    try:
        # 打开并调整大小
        img = Image.open(img_path)
        w, h = img.size
        scale = target_width / w
        new_h = int(h * scale)
        img_resized = img.resize((target_width, new_h), Image.LANCZOS)

        # 转换为小写 .jpg 扩展名
        basename = os.path.basename(img_path)
        new_name = os.path.splitext(basename)[0].lower() + ".jpg"
        img_resized.save(os.path.join(dst_dir, new_name), quality=95)

        if (i+1) % 100 == 0:
            print(f"Processed {i+1}/{len(img_files)} images")
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print(f"Done! Resized to {target_width}px width, saved as lowercase .jpg")
EOF
```

---

#### 步骤 2：创建配置文件

```bash
cd /home/songliyu/Documents/Depth-Anything-3
cat > da3_streaming/configs/dcim.yaml << 'EOF'
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'

Model:
  chunk_size: 25
  overlap: 12
  loop_chunk_size: 25
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
EOF
```

---

#### 步骤 3：执行流式预测

```bash
cd /home/songliyu/Documents/Depth-Anything-3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python da3_streaming/da3_streaming.py \
    --image_dir ./data/DCIM/images_resized \
    --config ./da3_streaming/configs/dcim.yaml \
    --output_dir ./output/dcim
```

**预期执行时间**：约 3-5 分钟（969帧）

---

### 四、预期输出

运行完成后，`./output/dcim/` 目录将包含：

| 文件 | 说明 |
|------|------|
| `camera_poses.txt` | 相机位姿（w2c格式） |
| `intrinsic.txt` | 相机内参 |
| `pcd/combined_pcd.ply` | **合并的彩色点云文件** |
| `camera_poses.ply` | 相机位姿点云 |
| `results_output/frame_*.npz` | 每帧的深度图、置信度、RGB、内参 |

---

### 五、可视化方案

创建 `visualize_dcim.py`：

```python
import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud("./output/dcim/pcd/combined_pcd.ply")
print(f"Loaded {len(pcd.points)} points")

# 居中显示
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
pcd.translate(-center)

# 修正上下颠倒
R = pcd.get_rotation_matrix_from_xyz([np.pi, 0, 0])
pcd.rotate(R, center=[0, 0, 0])

# 可视化
o3d.visualization.draw_geometries([pcd],
                                  window_name="DCIM 3D Reconstruction",
                                  width=1920,
                                  height=1080)
```

---

### 六、资源消耗预估

| 项目 | 预估值 |
|------|--------|
| **处理时间** | 3-5 分钟 |
| **显存峰值** | 13-14 GB |
| **磁盘占用** | 5-8 GB |

---

### 七、计划总结

| 步骤 | 操作 | 关键点 |
|------|------|--------|
| 1 | 图像预处理 | 降采样到640px + 转小写.jpg |
| 2 | 创建配置 | chunk_size=25, loop_enable=False |
| 3 | 执行预测 | 输出到 `./output/dcim` |
| 4 | 验证结果 | 检查点云文件 |
| 5 | 可视化 | Open3D 显示3D重建 |
