# iPhone 11 闭合路径视频 DA3-Streaming 流式预测计划

## 一、项目概述

| 项目 | 详情 |
|------|------|
| **视频文件** | `./data/IMG_7375.MOV` |
| **文件大小** | 1.2 GB |
| **视频时长** | 21 分 14 秒 (1274.7 秒) |
| **原始分辨率** | **1920 x 1080** (16:9) |
| **原始帧率** | 29.97 fps |
| **总帧数** | 约 **38,191 帧** |
| **拍摄路径** | **闭合路径**（用户确认） |
| **GPU 显存** | NVIDIA RTX 5060 Ti 16GB (16311 MiB) |
| **工作目录** | `/home/songliyu/Documents/Depth-Anything-3` |
| **输出目录** | `./output/iphone11-IMG_7375` |

---

## 二、视频信息分析

### 2.1 实际视频参数（通过 ffprobe 获取）

```
文件路径: ./data/IMG_7375.MOV
编码格式: HEVC (H.265)
分辨率:   1920 x 1080 (16:9)
帧率:     29.97 fps (30000/1001)
时长:     1274.675 秒 ≈ 21.24 分钟
总帧数:   ≈ 38,191 帧
码率:     7.9 Mbps
```

### 2.2 采样策略

| 采样率 | 抽取帧数 | 预估处理时间 | 说明 |
|--------|----------|--------------|------|
| 1 fps | ~1,275 帧 | ~5-8 分钟 | 帧稀疏，处理快 |
| 2 fps | ~2,549 帧 | ~10-15 分钟 | **推荐**，平衡质量和速度 |
| 3 fps | ~3,824 帧 | ~15-20 分钟 | 帧密集，质量高 |

**推荐采样率：2 fps**，约 2,549 帧

### 2.3 降采样分辨率

| 图像宽度 | 对应高度 (16:9) | 说明 |
|----------|-----------------|------|
| 480 px | 270 px | 保守配置 |
| 512 px | 288 px | 标准配置 |
| **560 px** | **315 px** | **推荐配置** |
| 600 px | 338 px | 高质量配置 |

---

## 三、现有数据分析回顾

### 3.1 South-Building 数据集测试结果（来自 `predict.md`）

| chunk_size | loop_enable | 图像分辨率 | 峰值显存 | 利用率 | 结果 |
|------------|-------------|------------|----------|--------|------|
| 30 | True | 512×384 | 15777 MB | 96.7% | ❌ OOM |
| 25 | True | 512×384 | 15730 MB | 96.4% | ❌ OOM |
| 22 | True | 512×384 | 15419 MB | 94.5% | ❌ OOM |
| 20 | True | 512×384 | 15743 MB | 96.5% | ❌ OOM（回环阶段） |
| **20** | **False** | **512×384** | **13703 MB** | **84.0%** | ✅ 成功 |

### 3.2 关键发现

1. **回环阶段显存激增**：回环检测时需要加载约 `2 × chunk_size` 的图像进行对比
2. **显存安全裕量**：16GB 显存在 `chunk_size=20, loop_enable=False` 时约 2.6GB 裕量
3. **闭合路径需求**：用户确认视频为闭合路径，回环检测对消除累积误差至关重要

### 3.3 显存使用分析

```
显存组成（预估）：
├── 模型权重（DA3 + SALAD）: ~7 GB
├── 图像数据（chunk_size × H × W × 3 × 4字节）: 可变
│   └── 560×315×3×4×18 ≈ 38 MB/chunk
├── 深度图/置信度: ~76 MB/chunk
├── 中间计算结果: 可变
└── 回环检测额外开销: ~2 GB（当启用时）
```

---

## 四、核心问题与解决方案

### 4.1 核心权衡

| 方案 | chunk_size | loop_enable | 图像分辨率 | 预估显存 | 优点 | 缺点 |
|------|------------|-------------|------------|----------|------|------|
| A | 20 | False | 512×288 | ~13.7 GB (84%) | 显存安全 | 闭合误差大 |
| B | 15 | True | 560×315 | ~13-14 GB (~80%) | 有回环优化 | chunk_size较小 |
| C | 18 | True | 560×315 | ~14-15 GB (~90%) | 平衡 | 显存接近上限 |
| D | 15 | True | 600×338 | ~14-15 GB (~90%) | 高分辨率 | 显存接近上限 |

### 4.2 推荐方案：**方案 C（激进平衡方案）**

**配置参数**：
- `chunk_size: 18`
- `overlap: 9`（约50%）
- `loop_enable: True`
- `loop_chunk_size: 18`
- 图像降采样宽度：**560 px** → **560×315**

**预估显存**：~14.5 GB (约89%)

**理由**：
1. chunk_size 18 相比 20 仅减少10%，但显著降低回环阶段OOM风险
2. 保留了回环检测功能，对闭合路径至关重要
3. 显存利用率接近90%，最大化利用硬件资源
4. 图像分辨率560px提供更多细节，提升重建质量
5. 16:9 比例保持原图宽高比，避免变形

---

## 五、对比实验设计

### 5.1 实验目的

验证不同配置在iPhone 11闭合路径视频上的表现，确定最优方案。

### 5.2 实验变量

| 变量 | 测试值 |
|------|--------|
| chunk_size | 12, 15, 18 |
| loop_enable | True, False |
| 图像宽度 | 512, 560, 600 |

### 5.3 实验矩阵

```
实验配置：
┌─────────┬───────────────┬──────────────┬──────────────┬──────────────┐
│ 实验组  │ chunk_size   │ loop_enable  │ 图像分辨率   │ 说明          │
├─────────┼───────────────┼──────────────┼──────────────┼──────────────┤
│ 对照组  │ 20           │ False        │ 512×288      │ 基准配置      │
│ 实验A   │ 12           │ True         │ 560×315      │ 保守配置      │
│ 实验B   │ 15           │ True         │ 560×315      │ 平衡配置      │
│ 实验C   │ 18           │ True         │ 560×315      │ ← 推荐配置    │
│ 实验D   │ 15           │ True         │ 600×338      │ 高分辨率测试  │
└─────────┴───────────────┴──────────────┴──────────────┴──────────────┘
```

### 5.4 评估指标

| 指标 | 说明 |
|------|------|
| **显存峰值** | nvidia-smi 监控 |
| **是否成功** | 是否完成全流程 |
| **处理速度** | fps（帧/秒） |
| **轨迹闭合度** | 优化前后首尾距离 |
| **重建质量** | 点云密度、一致性 |

---

## 六、详细执行步骤

### 步骤 0：环境准备

```bash
cd /home/songliyu/Documents/Depth-Anything-3

# 检查视频文件
ls -lh data/IMG_7375.MOV

# 检查模型权重
ls -lh weights/model.safetensors weights/dino_salad.ckpt weights/config.json

# 检查依赖
pip list | grep -E "(torch|safetensors|faiss|pypose)"

# 初始化 SALAD 子模块（如果未初始化）
git submodule update --init --recursive da3_streaming/loop_utils/salad
```

**预期输出**：
```
-rw-r--r-- 1 songliyu songliyu 1.2G data/IMG_7375.MOV
-rw-r--r-- 1 user user 6.8G weights/model.safetensors
-rw-r--r-- 1 user user 340M weights/dino_salad.ckpt
-rw-r--r-- 1 user user 3.0K weights/config.json
```

---

### 步骤 1：视频预处理（抽帧 + 降采样）

#### 1.1 验证视频信息

```bash
# 查看视频详细信息
ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,r_frame_rate -of default=noprint_wrappers=1 data/IMG_7375.MOV
```

**预期输出**：
```
width=1920
height=1080
r_frame_rate=30000/1001
duration=1274.675000
```

#### 1.2 视频抽帧

```bash
cd /home/songliyu/Documents/Depth-Anything-3

# 创建数据目录
mkdir -p data/iphone11-frames

# 抽帧：2 fps（每秒2帧），约生成 2549 帧
# 使用 -q:v 2 保证高质量JPEG输出
ffmpeg -i data/IMG_7375.MOV \
    -vf "fps=2" \
    -q:v 2 \
    -vsync 0 \
    data/iphone11-frames/frame_%06d.jpg

# 查看抽帧结果
echo "抽帧数量: $(ls data/iphone11-frames/*.jpg 2>/dev/null | wc -l)"
```

**预期输出**：
```
抽帧数量: 2549
```

#### 1.3 图像降采样脚本

```bash
cd /home/songliyu/Documents/Depth-Anything-3
mkdir -p data/iphone11-frames-560

python3 << 'EOF'
import os
import glob
from PIL import Image

src_dir = "data/iphone11-frames"
dst_dir = "data/iphone11-frames-560"
target_width = 560

os.makedirs(dst_dir, exist_ok=True)

# 获取所有图像文件
img_files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")) +
                   glob.glob(os.path.join(src_dir, "*.JPG")) +
                   glob.glob(os.path.join(src_dir, "*.png")))

print(f"Found {len(img_files)} images")

# 获取第一张图像尺寸，计算目标高度
if len(img_files) > 0:
    first_img = Image.open(img_files[0])
    w, h = first_img.size
    scale = target_width / w
    target_height = int(h * scale)
    print(f"Original resolution: {w} x {h}")
    print(f"Target resolution: {target_width} x {target_height}")
    print(f"Aspect ratio preserved: {w/h:.3f} -> {target_width/target_height:.3f}")

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
        print(f"Processed {i+1}/{len(img_files)} images ({100*(i+1)/len(img_files):.1f}%)")

print(f"Done! Resized to {target_width}px width, saved {len(img_files)} images")
EOF
```

**预期输出**：
```
Found 2549 images
Original resolution: 1920 x 1080
Target resolution: 560 x 315
Aspect ratio preserved: 1.778 -> 1.778
Processed 500/2549 images (19.6%)
Processed 1000/2549 images (39.2%)
Processed 1500/2549 images (58.8%)
Processed 2000/2549 images (78.5%)
Processed 2500/2549 images (98.1%)
Done! Resized to 560px width, saved 2549 images
```

---

### 步骤 2：创建实验配置文件

#### 2.1 创建配置目录

```bash
mkdir -p da3_streaming/configs/iphone11_exp
mkdir -p scripts
```

#### 2.2 创建对照组配置（chunk_size=20, loop_enable=False）

```bash
cat > da3_streaming/configs/iphone11_exp/control_chunk20_512_noloop.yaml << 'EOF'
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
EOF
```

#### 2.3 创建实验A配置（chunk_size=12）

```bash
cat > da3_streaming/configs/iphone11_exp/exp_a_chunk12_560.yaml << 'EOF'
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'

Model:
  chunk_size: 12
  overlap: 6
  loop_chunk_size: 12
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

#### 2.4 创建实验B配置（chunk_size=15）

```bash
cat > da3_streaming/configs/iphone11_exp/exp_b_chunk15_560.yaml << 'EOF'
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'

Model:
  chunk_size: 15
  overlap: 8
  loop_chunk_size: 15
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

#### 2.5 创建实验C配置（chunk_size=18，推荐配置）

```bash
cat > da3_streaming/configs/iphone11_exp/exp_c_chunk18_560.yaml << 'EOF'
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'

Model:
  chunk_size: 18
  overlap: 9
  loop_chunk_size: 18
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

#### 2.6 创建实验D配置（chunk_size=15, 600px高分辨率）

```bash
cat > da3_streaming/configs/iphone11_exp/exp_d_chunk15_600.yaml << 'EOF'
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'

Model:
  chunk_size: 15
  overlap: 8
  loop_chunk_size: 15
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

#### 2.7 创建600px降采样目录（用于实验D）

```bash
cd /home/songliyu/Documents/Depth-Anything-3
mkdir -p data/iphone11-frames-600

python3 << 'EOF'
import os
import glob
from PIL import Image

src_dir = "data/iphone11-frames"
dst_dir = "data/iphone11-frames-600"
target_width = 600

os.makedirs(dst_dir, exist_ok=True)

img_files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")) +
                   glob.glob(os.path.join(src_dir, "*.JPG")) +
                   glob.glob(os.path.join(src_dir, "*.png")))

print(f"Found {len(img_files)} images")
print(f"Resizing to {target_width}px width...")

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

print(f"Done! Resized to {target_width}px width, saved {len(img_files)} images")
EOF
```

---

### 步骤 3：创建辅助脚本

#### 3.1 显存监控脚本

```bash
cat > scripts/monitor_gpu.sh << 'EOF'
#!/bin/bash

# 显存监控脚本
# 使用方法: ./scripts/monitor_gpu.sh <实验名称>

LOG_FILE="gpu_memory_${1}.log"

echo "Timestamp, GPU, Utilization, Memory_Used, Memory_Total" > "$LOG_FILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    GPU_INFO=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits)
    echo "$TIMESTAMP, $GPU_INFO" >> "$LOG_FILE"
    sleep 1
done
EOF

chmod +x scripts/monitor_gpu.sh
```

#### 3.2 实验执行脚本

```bash
cat > scripts/run_experiments.sh << 'EOF'
#!/bin/bash

# iPhone 11 对比实验执行脚本

set -e

WORK_DIR="/home/songliyu/Documents/Depth-Anything-3"
IMAGE_DIR_560="$WORK_DIR/data/iphone11-frames-560"
IMAGE_DIR_600="$WORK_DIR/data/iphone11-frames-600"
IMAGE_DIR_512="$WORK_DIR/data/iphone11-frames"  # 原始帧（用于对照组）
CONFIG_DIR="$WORK_DIR/da3_streaming/configs/iphone11_exp"
OUTPUT_BASE="$WORK_DIR/output/iphone11-exp"

# 实验配置列表
declare -A EXPERIMENTS=(
    ["control"]="control_chunk20_512_noloop.yaml:$IMAGE_DIR_512:对照组-chunk20-noloop-512px"
    ["exp_a"]="exp_a_chunk12_560.yaml:$IMAGE_DIR_560:实验A-chunk12-loop-560px"
    ["exp_b"]="exp_b_chunk15_560.yaml:$IMAGE_DIR_560:实验B-chunk15-loop-560px"
    ["exp_c"]="exp_c_chunk18_560.yaml:$IMAGE_DIR_560:实验C-chunk18-loop-560px-推荐"
    ["exp_d"]="exp_d_chunk15_600.yaml:$IMAGE_DIR_600:实验D-chunk15-loop-600px"
)

cd "$WORK_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}iPhone 11 IMG_7375 对比实验执行${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}视频信息:${NC}"
echo -e "  文件: data/IMG_7375.MOV"
echo -e "  时长: 21分14秒 (1274.7秒)"
echo -e "  分辨率: 1920 x 1080"
echo ""

# 检查图像目录
echo -e "${BLUE}检查图像目录...${NC}"
if [ ! -d "$IMAGE_DIR_560" ]; then
    echo -e "${RED}错误: 560px 图像目录不存在: $IMAGE_DIR_560${NC}"
    echo -e "${YELLOW}请先执行视频抽帧和降采样步骤${NC}"
    exit 1
fi

IMG_COUNT_560=$(ls "$IMAGE_DIR_560"/*.jpg 2>/dev/null | wc -l)
echo -e "  560px 图像数量: ${GREEN}$IMG_COUNT_560${NC} 张"

if [ -d "$IMAGE_DIR_600" ]; then
    IMG_COUNT_600=$(ls "$IMAGE_DIR_600"/*.jpg 2>/dev/null | wc -l)
    echo -e "  600px 图像数量: ${GREEN}$IMG_COUNT_600${NC} 张"
fi

IMG_COUNT_ORIG=$(ls "$IMAGE_DIR_512"/*.jpg 2>/dev/null | wc -l)
echo -e "  原始帧数量: ${GREEN}$IMG_COUNT_ORIG${NC} 张"
echo ""

# 询问用户选择实验
echo -e "${BLUE}请选择要执行的实验：${NC}"
echo ""
for key in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r config_file img_dir desc <<< "${EXPERIMENTS[$key]}"
    if [ -d "$img_dir" ]; then
        echo -e "  ${GREEN}$key${NC}) $desc"
    else
        echo -e "  ${key}) $desc ${RED}[目录不存在]${NC}"
    fi
done
echo -e "  ${YELLOW}all${NC}) 执行所有可用实验"
echo -e "  ${YELLOW}quit${NC}) 退出"
echo ""
read -p "请输入选择: " choice

if [ "$choice" == "quit" ]; then
    echo "退出实验"
    exit 0
fi

# 执行实验函数
run_experiment() {
    local exp_key=$1
    local config_info=${EXPERIMENTS[$exp_key]}
    IFS=':' read -r config_file img_dir desc <<< "$config_info"

    local config_path="$CONFIG_DIR/$config_file"
    local output_dir="$OUTPUT_BASE/$exp_key"

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}执行实验: $desc${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "配置文件: $config_path"
    echo -e "图像目录: $img_dir"
    echo -e "输出目录: $output_dir"
    echo ""

    # 检查图像目录
    if [ ! -d "$img_dir" ]; then
        echo -e "${RED}错误: 图像目录不存在: $img_dir${NC}"
        return 1
    fi

    # 启动显存监控
    bash scripts/monitor_gpu.sh "$exp_key" &
    MONITOR_PID=$!
    echo -e "显存监控 PID: ${YELLOW}$MONITOR_PID${NC}"

    # 记录开始时间
    START_TIME=$(date +%s)

    # 执行预测
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python da3_streaming/da3_streaming.py \
        --image_dir "$img_dir" \
        --config "$config_path" \
        --output_dir "$output_dir" \
        2>&1 | tee "$output_dir/experiment_log.txt"

    # 记录结束时间
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # 停止显存监控
    kill $MONITOR_PID 2>/dev/null || true

    # 检查是否成功
    if [ -f "$output_dir/pcd/combined_pcd.ply" ]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}实验 $exp_key 完成！${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo -e "处理时间: ${YELLOW}$((DURATION / 60))分 $((DURATION % 60))秒${NC}"

        # 提取显存峰值
        if [ -f "gpu_memory_${exp_key}.log" ]; then
            PEAK_MEM=$(grep "Memory_Used" "gpu_memory_${exp_key}.log" | awk -F', ' '{print $3}' | sort -rn | head -1)
            echo -e "峰值显存: ${YELLOW}${PEAK_MEM} MB${NC}"
        fi
    else
        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}实验 $exp_key 失败！请检查日志${NC}"
        echo -e "${RED}========================================${NC}"
        return 1
    fi

    echo ""
}

# 执行选择
if [ "$choice" == "all" ]; then
    for key in "${!EXPERIMENTS[@]}"; do
        IFS=':' read -r config_file img_dir desc <<< "${EXPERIMENTS[$key]}"
        if [ -d "$img_dir" ]; then
            run_experiment "$key"
            sleep 5  # 冷却时间
        fi
    done
elif [ -n "${EXPERIMENTS[$choice]}" ]; then
    run_experiment "$choice"
else
    echo -e "${RED}无效选择: $choice${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "结果目录: $OUTPUT_BASE"
EOF

chmod +x scripts/run_experiments.sh
```

#### 3.3 单次实验快捷脚本

```bash
cat > scripts/run_single.sh << 'EOF'
#!/bin/bash

# 单次实验快捷脚本
# 使用方法: ./scripts/run_single.sh <实验名称>

EXP_KEY=$1

if [ -z "$EXP_KEY" ]; then
    echo "用法: ./scripts/run_single.sh <实验名称>"
    echo "可用实验: control, exp_a, exp_b, exp_c, exp_d"
    exit 1
fi

WORK_DIR="/home/songliyu/Documents/Depth-Anything-3"
IMAGE_DIR_560="$WORK_DIR/data/iphone11-frames-560"
IMAGE_DIR_600="$WORK_DIR/data/iphone11-frames-600"
IMAGE_DIR_512="$WORK_DIR/data/iphone11-frames"
CONFIG_DIR="$WORK_DIR/da3_streaming/configs/iphone11_exp"
OUTPUT_BASE="$WORK_DIR/output/iphone11-exp"

declare -A EXPERIMENTS=(
    ["control"]="control_chunk20_512_noloop.yaml:$IMAGE_DIR_512"
    ["exp_a"]="exp_a_chunk12_560.yaml:$IMAGE_DIR_560"
    ["exp_b"]="exp_b_chunk15_560.yaml:$IMAGE_DIR_560"
    ["exp_c"]="exp_c_chunk18_560.yaml:$IMAGE_DIR_560"
    ["exp_d"]="exp_d_chunk15_600.yaml:$IMAGE_DIR_600"
)

if [ -z "${EXPERIMENTS[$EXP_KEY]}" ]; then
    echo "错误: 未知实验名称 '$EXP_KEY'"
    exit 1
fi

IFS=':' read -r config_file img_dir <<< "${EXPERIMENTS[$EXP_KEY]}"

config_path="$CONFIG_DIR/$config_file"
output_dir="$OUTPUT_BASE/$EXP_KEY"

echo "执行实验: $EXP_KEY"
echo "配置: $config_path"
echo "图像: $img_dir"
echo "输出: $output_dir"
echo ""

cd "$WORK_DIR"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python da3_streaming/da3_streaming.py \
    --image_dir "$img_dir" \
    --config "$config_path" \
    --output_dir "$output_dir"
EOF

chmod +x scripts/run_single.sh
```

---

### 步骤 4：执行对比实验

#### 4.1 实验执行顺序

```
步骤：
1. 先执行对照组 (control) → 建立基准
2. 再执行实验A (exp_a) → 验证小chunk可行性
3. 再执行实验B (exp_b) → 平衡测试
4. 最后执行实验C (exp_c) → 推荐配置验证
5. 可选：实验D (exp_d) → 高分辨率测试
```

#### 4.2 使用自动化脚本

```bash
# 方式1: 使用交互式脚本
cd /home/songliyu/Documents/Depth-Anything-3
./scripts/run_experiments.sh

# 方式2: 直接执行单个实验
./scripts/run_single.sh exp_c
```

#### 4.3 手动执行命令

```bash
# 实验C（推荐配置）
cd /home/songliyu/Documents/Depth-Anything-3

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python da3_streaming/da3_streaming.py \
    --image_dir ./data/iphone11-frames-560 \
    --config ./da3_streaming/configs/iphone11_exp/exp_c_chunk18_560.yaml \
    --output_dir ./output/iphone11-exp/exp_c
```

---

### 步骤 5：结果分析与对比

#### 5.1 快速结果对比

```bash
cat > scripts/compare_results.sh << 'EOF'
#!/bin/bash

# 快速结果对比脚本

OUTPUT_BASE="./output/iphone11-exp"

echo "========================================="
echo "实验结果快速对比"
echo "========================================="
echo ""
printf "%-12s %-10s %-15s %-10s\n" "实验" "状态" "输出文件" "点云点数"
echo "-----------------------------------------"

for exp_dir in "$OUTPUT_BASE"/*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        ply_file="$exp_dir/pcd/combined_pcd.ply"

        if [ -f "$ply_file" ]; then
            # 获取点云点数（简化版）
            status="成功"
            output="combined_pcd.ply"
            points="N/A"
        else
            status="失败"
            output="N/A"
            points="N/A"
        fi

        printf "%-12s %-10s %-15s %-10s\n" "$exp_name" "$status" "$output" "$points"
    fi
done

echo "========================================="
EOF

chmod +x scripts/compare_results.sh

# 执行对比
./scripts/compare_results.sh
```

#### 5.2 详细结果分析（Python）

```bash
cat > scripts/analyze_results.py << 'EOF'
#!/usr/bin/env python3
"""
实验结果分析脚本
对比不同配置的显存使用、处理速度、重建质量
"""

import os
import re
import numpy as np
from pathlib import Path

def parse_gpu_log(log_file):
    """解析显存监控日志"""
    memory_used = []

    if not os.path.exists(log_file):
        return None, None

    with open(log_file, 'r') as f:
        for line in f:
            if 'Memory_Used' in line:
                parts = line.strip().split(', ')
                if len(parts) >= 4:
                    try:
                        memory_used.append(int(parts[2]))
                    except:
                        pass

    if memory_used:
        return max(memory_used), np.mean(memory_used)
    return None, None

def parse_experiment_log(log_file):
    """解析实验日志"""
    data = {
        'total_frames': 0,
        'loop_count': 0,
    }

    if not os.path.exists(log_file):
        return data

    with open(log_file, 'r') as f:
        content = f.read()

        # 提取帧数
        match = re.search(r'Found (\d+) images', content)
        if match:
            data['total_frames'] = int(match.group(1))

        # 提取回环数量
        match = re.search(r'Found (\d+) loop pairs', content)
        if match:
            data['loop_count'] = int(match.group(1))

    return data

def compare_experiments(output_base):
    """对比所有实验结果"""

    experiments = {
        'control': {'name': '对照组: chunk=20, loop=False, 512px', 'config': 'control_chunk20_512_noloop.yaml'},
        'exp_a': {'name': '实验A: chunk=12, loop=True, 560px', 'config': 'exp_a_chunk12_560.yaml'},
        'exp_b': {'name': '实验B: chunk=15, loop=True, 560px', 'config': 'exp_b_chunk15_560.yaml'},
        'exp_c': {'name': '实验C: chunk=18, loop=True, 560px (推荐)', 'config': 'exp_c_chunk18_560.yaml'},
        'exp_d': {'name': '实验D: chunk=15, loop=True, 600px', 'config': 'exp_d_chunk15_600.yaml'},
    }

    results = {}

    for exp_key, exp_info in experiments.items():
        exp_dir = os.path.join(output_base, exp_key)
        gpu_log = os.path.join(output_base, f'gpu_memory_{exp_key}.log')
        exp_log = os.path.join(exp_dir, 'experiment_log.txt')

        if os.path.exists(exp_dir):
            peak_mem, avg_mem = parse_gpu_log(gpu_log)
            exp_data = parse_experiment_log(exp_log)

            results[exp_key] = {
                'name': exp_info['name'],
                'config': exp_info['config'],
                'peak_memory_mb': peak_mem,
                'avg_memory_mb': avg_mem,
                'success': os.path.exists(os.path.join(exp_dir, 'pcd/combined_pcd.ply')),
                **exp_data
            }

    # 打印对比表
    print("\n" + "="*100)
    print("实验结果对比表")
    print("="*100)
    print(f"{'实验':<10} {'配置文件':<35} {'峰值显存':<12} {'平均显存':<12} {'帧数':<8} {'回环数':<8} {'状态':<8}")
    print("-"*100)

    for exp_key in ['control', 'exp_a', 'exp_b', 'exp_c', 'exp_d']:
        if exp_key in results:
            r = results[exp_key]
            status = "成功" if r['success'] else "失败"
            peak = f"{r['peak_memory_mb']} MB" if r['peak_memory_mb'] else "N/A"
            avg = f"{r['avg_memory_mb']:.0f} MB" if r['avg_memory_mb'] else "N/A"
            print(f"{exp_key:<10} {r['config']:<35} {peak:<12} {avg:<12} {r['total_frames']:<8} {r['loop_count']:<8} {status:<8}")

    print("="*100)

    # 打印推荐
    print("\n推荐配置:")
    successful = [k for k, v in results.items() if v['success'] and v['loop_count'] > 0]
    if successful:
        # 选择显存利用率最高的成功实验
        best = max(successful, key=lambda k: results[k]['peak_memory_mb'] or 0)
        print(f"  {results[best]['name']}")
        print(f"  峰值显存: {results[best]['peak_memory_mb']} MB")
    else:
        print("  暂无成功实验，请检查日志")

    return results

if __name__ == "__main__":
    output_base = "./output/iphone11-exp"
    results = compare_experiments(output_base)
EOF

chmod +x scripts/analyze_results.py

# 执行分析
python3 scripts/analyze_results.py
```

---

## 七、意外情况与解决方案

### 7.1 显存溢出 (OOM)

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| CUDA out of memory | chunk_size 过大 | 降低 chunk_size 到 12 或 10 |
| 回环阶段 OOM | 回环检测时显存激增 | 禁用回环检测或降低图像分辨率 |
| 模型加载失败 | 权重文件损坏 | 重新下载权重文件 |

**应急方案**：

```bash
# 方案1: 降低 chunk_size
# 修改配置文件中的 chunk_size 从 18 降至 12 或 10

# 方案2: 降低图像分辨率
# 将图像宽度从 560px 降至 512px 或 480px

# 方案3: 禁用回环检测
# 将 loop_enable 设为 False

# 方案4: 降低 overlap
# 将 overlap 从 chunk_size/2 降至 chunk_size/4
```

### 7.2 视频抽帧问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 抽帧数量异常 | 帧率设置错误 | 调整 `-vf "fps=N"` 参数 |
| 图像损坏 | 视频编码问题 | 使用 `-c:v mjpeg` 重新编码 |
| 抽帧失败 | ffmpeg 未安装 | `sudo apt install ffmpeg` |

### 7.3 回环检测失败

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| Found 0 loop pairs | 相似度阈值过高 | 降低 `similarity_threshold` 到 0.80 |
| 回环数量过多 | 相似度阈值过低 | 提高 `similarity_threshold` 到 0.90 |
| SALAD 模型加载失败 | 权重文件缺失 | 检查 `weights/dino_salad.ckpt` |

### 7.4 点云质量问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 点云稀疏 | 采样率过高 | 降低 `sample_ratio` 到 0.02 |
| 点云噪声大 | 置信度阈值过低 | 提高 `conf_threshold_coef` 到 0.85 |
| 点云为空 | 处理失败 | 检查实验日志 |

### 7.5 其他问题

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| ModuleNotFoundError | SALAD 子模块未初始化 | `git submodule update --init --recursive da3_streaming/loop_utils/salad` |
| 权限错误 | 脚本无执行权限 | `chmod +x scripts/*.sh` |
| 磁盘空间不足 | 临时文件过多 | 设置 `delete_temp_files: True` |

---

## 八、最终推荐配置

基于分析和实验设计，**推荐配置**为实验C：

```yaml
# 文件: da3_streaming/configs/iphone11_exp/exp_c_chunk18_560.yaml
Model:
  chunk_size: 18              # 平衡显存和处理效率
  overlap: 9                  # 50% 重叠
  loop_chunk_size: 18         # 与 chunk_size 相同
  loop_enable: True           # 闭合路径必须启用
  align_lib: 'triton'
  align_method: 'sim3'

Loop:
  SALAD:
    image_size: [336, 336]
    similarity_threshold: 0.85   # 可根据回环数量调整（0.80-0.90）
    top_k: 5
    use_nms: True
    nms_threshold: 25
```

**如果实验C失败，降级方案**：
1. **实验B** (chunk_size=15): 显存更安全
2. **实验A** (chunk_size=12): 最保守配置

---

## 九、预期输出

运行成功后，输出目录将包含：

| 文件/目录 | 说明 |
|-----------|------|
| `camera_poses.txt` | 相机位姿（每行12个参数，w2c格式） |
| `intrinsic.txt` | 相机内参（每行 fx, fy, cx, cy） |
| `pcd/combined_pcd.ply` | **合并的彩色点云文件** |
| `pcd/0_pcd.ply, 1_pcd.ply, ...` | 各chunk的点云文件 |
| `camera_poses.ply` | 相机位姿点云（轨迹可视化） |
| `sim3_opt_result.png` | **回环优化前后轨迹对比图** |
| `results_output/frame_*.npz` | 每帧的深度图、置信度、RGB图像、内参 |
| `experiment_log.txt` | 实验执行日志 |

---

## 十、结果可视化

### 10.1 查看轨迹优化图

```bash
# 使用系统默认图片查看器
xdg-open output/iphone11-exp/exp_c/sim3_opt_result.png
```

### 10.2 查看3D点云

```bash
# 使用 MeshLab
meshlab output/iphone11-exp/exp_c/pcd/combined_pcd.ply

# 或使用 CloudCompare
cloudcompare output/iphone11-exp/exp_c/pcd/combined_pcd.ply
```

### 10.3 使用 Open3D 可视化

```python
import open3d as o3d
import numpy as np

# 读取点云
pcd = o3d.io.read_point_cloud("output/iphone11-exp/exp_c/pcd/combined_pcd.ply")
print(f"Loaded {len(pcd.points)} points")

# 居中显示
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
pcd.translate(-center)

# 可视化
o3d.visualization.draw_geometries(
    [pcd],
    window_name="iPhone 11 IMG_7375 3D Reconstruction",
    width=1920,
    height=1080
)
```

---

## 十一、执行清单

- [ ] 步骤 0: 环境准备（检查权重、依赖）
- [ ] 步骤 1: 视频抽帧（ffmpeg, 2 fps）
- [ ] 步骤 1: 图像降采样（560px 和 600px）
- [ ] 步骤 2: 创建实验配置文件（5个）
- [ ] 步骤 3: 创建辅助脚本（监控、执行）
- [ ] 步骤 4: 执行对比实验
- [ ] 步骤 5: 结果分析与对比
- [ ] 步骤 6: 可视化最终点云

---

## 十二、时间预估

| 步骤 | 预估时间 |
|------|----------|
| 视频抽帧 (2fps) | 5-10 分钟 |
| 图像降采样 (560px) | 2-3 分钟 |
| 图像降采样 (600px, 可选) | 2-3 分钟 |
| 单次实验（~2549帧） | 10-15 分钟 |
| 全部实验（5组） | 50-75 分钟 |
| 结果分析 | 5 分钟 |
| **总计** | **约 1-1.5 小时** |

---

## 十三、快速开始指南

如果你想快速开始，可以直接执行推荐配置：

```bash
# 1. 进入项目目录
cd /home/songliyu/Documents/Depth-Anything-3

# 2. 视频抽帧（如果还没做）
mkdir -p data/iphone11-frames
ffmpeg -i data/IMG_7375.MOV -vf "fps=2" -q:v 2 -vsync 0 data/iphone11-frames/frame_%06d.jpg

# 3. 图像降采样
python3 -c "
import os, glob
from PIL import Image
os.makedirs('data/iphone11-frames-560', exist_ok=True)
img_files = sorted(glob.glob('data/iphone11-frames/*.jpg'))
print(f'Processing {len(img_files)} images...')
for i, img_path in enumerate(img_files):
    img = Image.open(img_path)
    w, h = img.size
    new_h = int(h * 560 / w)
    img_resized = img.resize((560, new_h), Image.LANCZOS)
    basename = os.path.basename(img_path)
    img_resized.save(f'data/iphone11-frames-560/{os.path.splitext(basename)[0].lower()}.jpg')
    if (i+1) % 500 == 0:
        print(f'Progress: {i+1}/{len(img_files)}')
print('Done!')
"

# 4. 创建推荐配置
mkdir -p da3_streaming/configs/iphone11_exp
# (使用步骤2.5中的配置内容)

# 5. 执行预测
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python da3_streaming/da3_streaming.py \
    --image_dir ./data/iphone11-frames-560 \
    --config ./da3_streaming/configs/iphone11_exp/exp_c_chunk18_560.yaml \
    --output_dir ./output/iphone11-IMG_7375
```

---

*本计划文档生成时间: 2026-01-21*
*视频文件: ./data/IMG_7375.MOV*
*项目路径: /home/songliyu/Documents/Depth-Anything-3*
