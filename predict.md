# South-Building 数据集 DA3-Streaming 流式预测执行记录

## 环境信息

| 项目 | 详情 |
|------|------|
| **数据集** | `data/south-building.zip` |
| **图像数量** | 128张 |
| **原始分辨率** | 3072 x 2304 |
| **文件扩展名** | `.JPG`（大写） |
| **GPU** | NVIDIA RTX 5060 Ti 16GB (16311 MiB) |
| **工作目录** | `/home/songliyu/Documents/Depth-Anything-3` |
| **输出目录** | `./output/south-building` |

---

## 执行步骤

### 步骤 1：解压数据集

```bash
cd /home/songliyu/Documents/Depth-Anything-3
unzip -q data/south-building.zip -d data/
```

解压后结构：
```
data/
└── south-building/
    ├── database.db
    └── images/
        └── *.JPG (128张图像)
```

---

### 步骤 2：下载模型权重

```bash
cd /home/songliyu/Documents/Depth-Anything-3
bash ./da3_streaming/scripts/download_weights.sh
```

下载内容（约 7.1GB）：
- `weights/dino_salad.ckpt` (~340 MB)
- `weights/config.json` (~3 KB)
- `weights/model.safetensors` (~6.76 GB)

**注意**：下载过程中遇到网络错误，使用 curl 断点续传完成下载。

---

### 步骤 3：安装依赖

```bash
cd /home/songliyu/Documents/Depth-Anything-3
pip install -r requirements.txt
pip install -r da3_streaming/requirements.txt
```

---

### 步骤 4：初始化 SALAD git 子模块

```bash
git submodule update --init --recursive da3_streaming/loop_utils/salad
```

**问题**：执行 `da3_streaming.py` 时出现 `ModuleNotFoundError: No module named 'loop_utils.salad.models'` 错误，原因是 SALAD 子模块未初始化。

---

### 步骤 5：图像预处理（降采样 + 扩展名转换）

**原因**：
1. 原始分辨率 3072x2304 过高，会导致显存溢出
2. 代码只支持小写 `.jpg`/`.png` 扩展名

```bash
cd /home/songliyu/Documents/Depth-Anything-3
mkdir -p data/south-building/images_resized

python3 << 'EOF'
import os
import glob
from PIL import Image

src_dir = "data/south-building/images"
dst_dir = "data/south-building/images_resized"
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

---

### 步骤 6：创建配置文件

创建 `da3_streaming/configs/south_building.yaml`：

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

---

### 步骤 7：执行流式预测（显存优化测试）

#### 7.1 显存测试结果

| chunk_size | loop_enabled | 峰值显存 | 利用率 | 结果 |
|------------|--------------|----------|--------|------|
| 30 | True | 15777 MB | 96.7% | ❌ OOM |
| 25 | True | 15730 MB | 96.4% | ❌ OOM |
| 22 | True | 15419 MB | 94.5% | ❌ OOM |
| 20 | True | 15743 MB | 96.5% | ❌ OOM (回环阶段) |
| **20** | **False** | **13703 MB** | **84.0%** | ✅ **成功** |

**结论**：回环检测阶段需要加载约 2×chunk_size 的图像（约40张），导致显存翻倍。对于16GB显存，禁用回环检测后 chunk_size=20 是安全配置，显存利用率 84%。

#### 7.2 最终执行命令

```bash
cd /home/songliyu/Documents/Depth-Anything-3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python da3_streaming/da3_streaming.py \
    --image_dir ./data/south-building/images_resized \
    --config ./da3_streaming/configs/south_building.yaml \
    --output_dir ./output/south-building
```

---

## 输出结果

运行完成后，`./output/south-building/` 目录包含：

| 文件 | 说明 |
|------|------|
| `camera_poses.txt` | 相机位姿（每行12个参数，3x4外参矩阵，w2c格式） |
| `intrinsic.txt` | 相机内参（每行 fx, fy, cx, cy） |
| `pcd/combined_pcd.ply` | **合并的彩色点云文件** (397,588 个点) |
| `pcd/0_pcd.ply, 1_pcd.ply, ...` | 各chunk的点云文件 |
| `camera_poses.ply` | 相机位姿点云 |
| `results_output/frame_*.npz` | 每帧的深度图、置信度、RGB图像、内参 |

---

## 可视化重建结果

### 问题1：项目本身不支持交互式3D可视化

**解决方案**：使用 Open3D 创建独立可视化脚本

### 问题2：点云上下颠倒

**解决方案**：绕 X 轴旋转 180 度修正方向

### 可视化代码

```python
import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud("./output/south-building/pcd/combined_pcd.ply")
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
    window_name="South-Building 3D Reconstruction (Corrected)",
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

## 执行总结

| 步骤 | 操作 | 状态 |
|------|------|------|
| 1 | 解压数据集 | ✅ |
| 2 | 下载模型权重 | ✅ |
| 3 | 安装依赖 | ✅ |
| 4 | 初始化 SALAD 子模块 | ✅ |
| 5 | 图像预处理 | ✅ |
| 6 | 创建配置文件 | ✅ |
| 7 | 显存优化测试 | ✅ |
| 8 | 执行流式预测 | ✅ |
| 9 | 可视化重建结果 | ✅ |

**最终配置**：
- chunk_size: 20
- loop_enable: False
- 峰值显存: 13703 MB / 16311 MB (84.0%)
- 点云数量: 397,588 个点
