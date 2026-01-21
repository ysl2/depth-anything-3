# iPhone 11 预测质量改进计划

## 一、当前问题诊断

### 1.1 关键问题

| 问题 | 当前配置 | 影响 |
|------|----------|------|
| **帧采样率过低** | 2fps → 2549帧 (仅6.7%) | 帧间间隔大，运动估计不准确 |
| **图像分辨率过低** | 560px (仅29%原图) | 特征检测质量差，深度估计精度低 |
| **回环检测不足** | 仅4对回环 | 闭合路径优化不充分 |
| **点云稀疏** | sample_ratio=0.015 | 重建细节不足 |

### 1.2 质量对比

```
原视频:      1920×1080 @ 30fps = 38,191帧
当前配置:    560×995 @ 2fps   = 2,549帧  (6.7%帧数, 29%分辨率)
```

---

## 二、改进方案

### 方案 A: 激进高质量方案

**目标**: 最大化重建质量，显存接近上限

| 参数 | 当前值 | 改进值 | 说明 |
|------|--------|--------|------|
| 帧采样率 | 2fps | **5fps** | ~6,373帧，提升2.5倍 |
| 图像宽度 | 560px | **720px** | 保持更多细节 |
| chunk_size | 18 | **15** | 降低以容纳更多帧 |
| overlap | 9 | **8** | 适度重叠 |
| loop_threshold | 0.85 | **0.80** | 检测更多回环 |
| keypoint_num | 5000 | **8000** | 更多特征点 |
| sample_ratio | 0.015 | **0.025** | 更密集点云 |

**预估显存**: ~14-15 GB

### 方案 B: 平衡质量方案 (推荐)

**目标**: 质量与稳定性的最佳平衡

| 参数 | 当前值 | 改进值 | 说明 |
|------|--------|--------|------|
| 帧采样率 | 2fps | **4fps** | ~5,099帧，提升2倍 |
| 图像宽度 | 560px | **640px** | 提升分辨率 |
| chunk_size | 18 | **16** | 平衡配置 |
| overlap | 9 | **8** | 适度重叠 |
| loop_threshold | 0.85 | **0.82** | 适中回环检测 |
| keypoint_num | 5000 | **7000** | 更多特征点 |
| sample_ratio | 0.015 | **0.020** | 适度密集点云 |

**预估显存**: ~13-14 GB

### 方案 C: 保守质量方案

**目标**: 稳定优先，适度提升

| 参数 | 当前值 | 改进值 | 说明 |
|------|--------|--------|------|
| 帧采样率 | 2fps | **3fps** | ~3,824帧，提升50% |
| 图像宽度 | 560px | **600px** | 适度提升 |
| chunk_size | 18 | **16** | 安全配置 |
| overlap | 9 | **8** | 适度重叠 |
| loop_threshold | 0.85 | **0.82** | 适中回环检测 |
| keypoint_num | 5000 | **6000** | 略多特征点 |
| sample_ratio | 0.015 | **0.018** | 略密集点云 |

**预估显存**: ~12-13 GB

---

## 三、推荐执行: 方案 B (平衡质量方案)

### 3.1 新配置参数

```yaml
# exp_improved.yaml
Model:
  chunk_size: 16
  overlap: 8
  loop_chunk_size: 16
  loop_enable: True
  ref_view_strategy: 'saddle_balanced'

  Sparse_Align:
    keypoint_select: 'orb'
    keypoint_num: 7000          # 提升至7000

  IRLS:
    delta: 0.1
    max_iters: 5
    tol: 1e-9

  Pointcloud_Save:
    sample_ratio: 0.020         # 提升至0.020
    conf_threshold_coef: 0.75

Loop:
  SALAD:
    similarity_threshold: 0.82  # 降低至0.82
    top_k: 8                    # 提升至8
    use_nms: True
    nms_threshold: 25
```

### 3.2 数据预处理

```bash
# 1. 重新抽帧: 4fps
mkdir -p data/iphone11-frames-4fps
ffmpeg -i data/IMG_7375.MOV \
    -vf "fps=4" \
    -q:v 2 \
    -vsync 0 \
    data/iphone11-frames-4fps/frame_%06d.jpg
# 预期: ~5,099帧

# 2. 降采样至 640px 宽度
mkdir -p data/iphone11-frames-640
python3 downscale.py \
    --input data/iphone11-frames-4fps \
    --output data/iphone11-frames-640 \
    --width 640
# 预期: 640×1140 (保持9:16比例)
```

### 3.3 预期改进

| 指标 | 当前 | 改进后 | 提升 |
|------|------|--------|------|
| 帧数 | 2,549 | ~5,099 | **+100%** |
| 分辨率 | 560×995 | 640×1140 | **+31%像素** |
| 特征点数 | 5000 | 7000 | **+40%** |
| 点云密度 | 1.5% | 2.0% | **+33%** |
| 回环检测 | 4对 | 预计8-12对 | **+100-200%** |

---

## 四、执行步骤

### 步骤 1: 重新抽帧 (4fps)
```bash
cd /home/songliyu/Documents/Depth-Anything-3
mkdir -p data/iphone11-frames-4fps
ffmpeg -i data/IMG_7375.MOV \
    -vf "fps=4" \
    -q:v 2 \
    -vsync 0 \
    data/iphone11-frames-4fps/frame_%06d.jpg
```

### 步骤 2: 降采样至 640px
```python
# 使用Python脚本降采样
python3 << 'EOF'
import os, glob
from PIL import Image

src_dir = "data/iphone11-frames-4fps"
dst_dir = "data/iphone11-frames-640"
target_width = 640

os.makedirs(dst_dir, exist_ok=True)
img_files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")))

print(f"Found {len(img_files)} images")

for i, img_path in enumerate(img_files):
    img = Image.open(img_path)
    w, h = img.size
    new_h = int(h * target_width / w)
    img_resized = img.resize((target_width, new_h), Image.LANCZOS)
    basename = os.path.basename(img_path)
    img_resized.save(os.path.join(dst_dir, basename.lower()))
    if (i+1) % 500 == 0:
        print(f"Processed {i+1}/{len(img_files)}")

print(f"Done! Saved {len(img_files)} images to {dst_dir}")
EOF
```

### 步骤 3: 创建改进配置
```bash
mkdir -p da3_streaming/configs/iphone11_exp
cat > da3_streaming/configs/iphone11_exp/exp_improved_640.yaml << 'EOF'
Weights:
  DA3: './weights/model.safetensors'
  DA3_CONFIG: './weights/config.json'
  SALAD: './weights/dino_salad.ckpt'

Model:
  chunk_size: 16
  overlap: 8
  loop_chunk_size: 16
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
    keypoint_num: 7000

  IRLS:
    delta: 0.1
    max_iters: 5
    tol: 1e-9

  Pointcloud_Save:
    sample_ratio: 0.020
    conf_threshold_coef: 0.75

Loop:
  SALAD:
    image_size: [336, 336]
    batch_size: 32
    similarity_threshold: 0.82
    top_k: 8
    use_nms: True
    nms_threshold: 25

  SIM3_Optimizer:
    lang_version: 'cpp'
    max_iterations: 30
    lambda_init: 1e-6
EOF
```

### 步骤 4: 执行预测
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python da3_streaming/da3_streaming.py \
    --image_dir ./data/iphone11-frames-640 \
    --config ./da3_streaming/configs/iphone11_exp/exp_improved_640.yaml \
    --output_dir ./output/iphone11-exp/improved
```

---

## 五、风险评估与应对

| 风险 | 概率 | 应对措施 |
|------|------|----------|
| OOM | 中 | 如发生，降至方案C (3fps, 600px) |
| 处理时间过长 | 低 | 预计20-25分钟，可接受 |
| 磁盘空间不足 | 低 | 开启delete_temp_files |

---

## 六、对比实验

为了验证改进效果，建议对比:

| 实验组 | 帧率 | 分辨率 | 说明 |
|--------|------|--------|------|
| 当前 (exp_c) | 2fps | 560px | 基准 |
| 改进 (improved) | 4fps | 640px | 推荐 |
| 激进 (aggressive) | 5fps | 720px | 如改进成功 |

---

*生成时间: 2026-01-21*
*状态: 待用户审批*
