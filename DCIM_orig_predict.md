# DCIM 数据集 (DJI Mini3 Pro) DA3-Streaming 执行日志

## 环境信息

| 项目 | 详情 |
|------|------|
| **数据集** | `./data/DCIM/images/` (DJI Mini3 Pro 航拍图像) |
| **图像数量** | 969张 |
| **原始分辨率** | 4032 x 3024 |
| **预处理分辨率** | 640 x 480 (降采样) |
| **GPU** | NVIDIA RTX 5060 Ti 16GB |
| **工作目录** | `/home/songliyu/Documents/Depth-Anything-3` |
| **输出目录** | `./output/dcim` |

---

## 执行步骤

### 步骤 1：图像预处理

```bash
mkdir -p data/DCIM/images_resized
python3 预处理脚本...
```

**结果**：
- 969张图像全部处理
- 降采样到 640px 宽度
- 转换为小写 `.jpg` 扩展名
- 保存到 `data/DCIM/images_resized/`

---

### 步骤 2：创建配置文件

配置文件 `da3_streaming/configs/dcim.yaml`：

```yaml
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
  depth_threshold: 15.0

  save_depth_conf_result: True
  save_debug_info: False

  Pointcloud_Save:
    sample_ratio: 0.01
    conf_threshold_coef: 0.75
```

---

### 步骤 3：执行流式预测

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python da3_streaming/da3_streaming.py \
    --image_dir ./data/DCIM/images_resized \
    --config ./da3_streaming/configs/dcim.yaml \
    --output_dir ./output/dcim
```

**执行结果**：
- 969张图像，分为74个chunks处理
- 每个chunk包含25张图像，重叠12张
- 处理时间：约 3-5 分钟
- 显存使用：安全范围内（chunk_size=25, loop_enable=False）

---

## 输出结果

运行完成后，`./output/dcim/` 目录包含：

| 文件 | 说明 |
|------|------|
| `camera_poses.txt` | 相机位姿（w2c格式，969行） |
| `intrinsic.txt` | 相机内参 |
| `pcd/combined_pcd.ply` | **合并的彩色点云文件 (2,536,530 个点)** |
| `camera_poses.ply` | 相机位姿点云 |
| `dcim.yaml` | 配置文件备份 |

**点云统计**：
- 总点数：**2,536,530**
- chunk数量：74
- 节省磁盘空间：9.78 GB

---

## GPU 优化效果

根据测试结果（参考south-building数据集经验）：

| chunk_size | loop_enable | 峰值显存 | DCIM结果 |
|------------|-------------|----------|----------|
| 30 | True | ~15.7 GB | ❌ OOM (参考) |
| 20 | False | ~13.7 GB | ✅ 成功 (参考) |
| **25** | **False** | **~14 GB** | ✅ **成功** |

**DCIM最终配置**：
- chunk_size: 25（在安全范围内最大化利用显存）
- loop_enable: False（避免回环检测显存翻倍）
- target_width: 640（平衡质量与速度）
- 预计峰值显存: ~14 GB

---

## 执行总结

| 步骤 | 操作 | 状态 |
|------|------|------|
| 1 | 图像预处理 | ✅ 969张图像处理完成 |
| 2 | 创建配置文件 | ✅ dcim.yaml |
| 3 | 执行流式预测 | ✅ 74个chunks处理完成 |
| 4 | 点云合并 | ✅ 2,536,530个点 |
| 5 | 磁盘清理 | ✅ 节省9.78 GB |

---

## 关键性能指标

| 指标 | 值 |
|------|-----|
| 处理图像数 | 969 |
| 处理chunks | 74 |
| chunk_size | 25 |
| 点云总点数 | 2,536,530 |
| 预估显存使用 | ~14 GB / 16 GB |
| 磁盘空间节省 | 9.78 GB |
