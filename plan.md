# 在 South-Building 数据集上执行 DA3-Streaming 流式预测
## 最终版计划

---

### 一、数据集与环境信息

| 项目 | 详情 |
|------|------|
| **数据集** | `data/south-building.zip` |
| **图像数量** | 128张 |
| **原始分辨率** | 3072 x 2304（需降采样） |
| **文件扩展名** | `.JPG`（大写，预处理时转小写） |
| **GPU** | NVIDIA RTX 5060 Ti 16GB |
| **工作目录** | `/home/songliyu/Documents/Depth-Anything-3` |
| **输出目录** | `./output/south-building` |

---

### 二、执行步骤

#### 步骤 1：解压数据集

```bash
cd /home/songliyu/Documents/Depth-Anything-3
unzip -q data/south-building.zip -d data/
```

解压后结构：
```
data/
└── south-building/
    ├── database.db          (COLMAP数据库，不使用)
    └── images/
        └── *.JPG            (128张图像)
```

---

#### 步骤 2：下载模型权重

```bash
cd /home/songliyu/Documents/Depth-Anything-3
bash ./da3_streaming/scripts/download_weights.sh
```

下载内容（约 7.1GB）：
- `weights/dino_salad.ckpt` (~340 MB)
- `weights/config.json` (~3 KB)
- `weights/model.safetensors` (~6.76 GB)

---

#### 步骤 3：安装依赖

```bash
cd /home/songliyu/Documents/Depth-Anything-3
pip install -r requirements.txt
pip install -r da3_streaming/requirements.txt
```

`da3_streaming/requirements.txt` 包含：
- faiss-gpu
- pandas
- prettytable
- einops
- safetensors
- numba
- pypose

**可选**：安装点云查看器（用于结果可视化）
```bash
sudo apt install meshlab
# 或
sudo apt install cloudcompare
```

---

#### 步骤 4：图像预处理（降采样 + 扩展名转换）

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
    # 打开并调整大小
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

预期输出：
```
Found 128 images
Processed 20/128 images
Processed 40/128 images
...
Done! Resized to 512px width, saved as lowercase .jpg
```

---

#### 步骤 5：创建配置文件

```bash
cd /home/songliyu/Documents/Depth-Anything-3
cat > da3_streaming/configs/south_building.yaml << 'EOF'
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'

Model:
  chunk_size: 40
  overlap: 20
  loop_chunk_size: 20
  loop_enable: True
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
EOF
```

---

#### 步骤 6：执行流式预测

```bash
cd /home/songliyu/Documents/Depth-Anything-3

python da3_streaming/da3_streaming.py \
    --image_dir ./data/south-building/images_resized \
    --config ./da3_streaming/configs/south_building.yaml \
    --output_dir ./output/south-building
```

---

### 三、预期输出

运行完成后，`./output/south-building/` 目录将包含：

| 文件 | 说明 | 可视化用途 |
|------|------|-----------|
| `camera_poses.txt` | 相机位姿（每行12个参数，3x4外参矩阵，w2c格式） | 用于重新生成点云 |
| `intrinsic.txt` | 相机内参（每行 fx, fy, cx, cy） | 相机参数参考 |
| `pcd/combined_pcd.ply` | **合并的彩色点云文件** | **3D场景重建可视化** |
| `camera_poses.ply` | **相机位姿点云** | **相机轨迹可视化** |
| `sim3_opt_result.png` | **回环优化前后轨迹对比图** | **直观查看优化效果** |
| `results_output/frame_*.npz` | 每帧的深度图、置信度、RGB图像、内参 | 单帧深度可视化 |

---

### 四、结果验证与可视化

#### 4.1 快速验证输出文件

```bash
# 检查输出文件是否完整
ls -lh output/south-building/
ls -lh output/south-building/pcd/
ls -lh output/south-building/results_output/ | head -20
```

#### 4.2 查看轨迹优化图（最直观）

```bash
# 使用系统默认图片查看器
xdg-open output/south-building/sim3_opt_result.png
# 或
eog output/south-building/sim3_opt_result.png
```

**说明**：此图显示相机轨迹在回环优化前后的对比
- **蓝色线**：优化前的相机轨迹
- **橙色线**：优化后的相机轨迹
- 轨迹闭合程度越好，说明重建效果越好

#### 4.3 查看3D点云重建结果

```bash
# 使用 MeshLab
meshlab output/south-building/pcd/combined_pcd.ply

# 或使用 CloudCompare
cloudcompare output/south-building/pcd/combined_pcd.ply
```

**操作提示**：
- 鼠标左键拖拽：旋转视角
- 鼠标滚轮：缩放
- 鼠标中键拖拽：平移
- 可以看到带颜色的3D场景重建结果

#### 4.4 查看单帧深度图（可选）

```python
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/songliyu/Documents/Depth-Anything-3/src')
from depth_anything_3.utils.visualize import visualize_depth

# 读取某一帧的结果
data = np.load("output/south-building/results_output/frame_0000.npz")

# 可视化深度图
depth_vis = visualize_depth(data["depth"], cmap="Spectral")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(data["image"])
plt.title("RGB Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth_vis)
plt.title("Depth Map")
plt.axis("off")

plt.tight_layout()
plt.savefig("frame_0000_comparison.png", dpi=150)
plt.show()
```

---

### 五、结果可视化详解

#### 5.1 可视化工具对比

| 工具 | 安装方式 | 优点 | 适用场景 |
|------|----------|------|----------|
| **图片查看器** | 系统自带 | 最简单快捷 | 快速查看轨迹优化图 |
| **MeshLab** | `sudo apt install meshlab` | 功能全面，开源 | 查看/编辑点云 |
| **CloudCompare** | `sudo apt install cloudcompare` | 专业点云处理 | 点云分析和编辑 |
| **Open3D** | `pip install open3d` | Python可编程 | 自定义可视化脚本 |

#### 5.2 使用 Open3D 可视化（Python）

```bash
# 安装 Open3D
pip install open3d
```

```python
import open3d as o3d

# 读取点云
pcd = o3d.io.read_point_cloud("output/south-building/pcd/combined_pcd.ply")

# 可视化
o3d.visualization.draw_geometries([pcd],
                                  window_name="South-Building Point Cloud",
                                  width=1920,
                                  height=1080)
```

#### 5.3 重新生成点云（可选）

如果需要调整置信度阈值或采样率重新生成点云：

```bash
cd /home/songliyu/Documents/Depth-Anything-3

python da3_streaming/npz_output_process.py \
    --npz_folder ./output/south-building/results_output \
    --pose_file ./output/south-building/camera_poses.txt \
    --output_file ./output/south-building/output_custom.ply \
    --conf_threshold_coef 0.75 \
    --sample_ratio 0.015
```

**参数说明**：
- `--conf_threshold_coef`：置信度阈值系数（默认0.5，越高过滤越严格）
- `--sample_ratio`：采样率（默认0.015，即1.5%的点被保留，值越小点云越稀疏）

---

### 六、资源消耗预估

| 项目 | 预估值 |
|------|--------|
| **处理时间** | 15-30 秒（128帧 @ ~8 FPS） |
| **显存峰值** | 12-15 GB |
| **磁盘占用** | 1-2 GB |

---

### 七、故障排除

| 问题 | 解决方案 |
|------|----------|
| 显存不足 | 降低 `chunk_size` 到 30，或降低图像分辨率到 384px |
| 找不到图像 | 确认 `images_resized` 目录中有 `.jpg` 小写文件 |
| 权重文件缺失 | 运行 `bash ./da3_streaming/scripts/download_weights.sh` |
| 依赖缺失 | 运行 `pip install -r da3_streaming/requirements.txt` |
| 点云查看器未安装 | 运行 `sudo apt install meshlab` 或 `sudo apt install cloudcompare` |

---

### 八、计划总结

| 步骤 | 操作 | 关键点 |
|------|------|--------|
| 1 | 解压数据集 | 生成 `data/south-building/images/` |
| 2 | 下载权重 | 约 7.1GB，首次运行需要 |
| 3 | 安装依赖 | 两个 requirements.txt + 可选安装点云查看器 |
| 4 | 图像预处理 | **降采样到512px + 转小写.jpg** |
| 5 | 创建配置 | chunk_size=40 适配16GB显存 |
| 6 | 执行预测 | 输出到 `./output/south-building` |
| 7 | 验证与可视化 | 检查文件、查看轨迹图、查看点云 |
