# FastSAM + DA3-Streaming 语义分割集成实现计划

**创建时间**: 2026年02月07日 18:17
**目标**: 在DA3-Streaming中集成FastSAM语义分割，实现长视频的三维语义重建可视化

---

## 项目概述

### 目标
将FastSAM轻量级语义分割模型集成到DA3-Streaming中，实现：
1. 长视频的深度估计 + 相机位姿估计 + 语义分割
2. 生成带语义标签的三维点云
3. 支持语义可视化

### 硬件约束
- **显存**: 16GB
- **磁盘**: 1TB
- **要求**: Training-free，无需额外训练数据

### 技术选型
| 组件 | 模型 | 显存需求 | 说明 |
|------|------|---------|------|
| 深度估计 | DA3 | ~3GB | Depth Anything 3 |
| 语义分割 | FastSAM-x | ~2.6GB | 比SAM快50倍 |
| 总显存 | - | ~6GB | 交替使用，不会同时占用 |

---

## 文件结构

### 新增文件

```
da3_streaming/
├── semantic_segmentation.py    # FastSAM语义分割模块
├── semantic_ply.py                # 扩展PLY格式支持语义标签
└── da3_semantic_streaming.py     # 集成的语义增强DA3-Streaming

configs/
└── base_config_semantic_16gb.yaml # 16GB语义分割配置文件

plans/
└── fastsam_integration_plan_20260207_1817.md  # 本文件
```

---

## 实现进度

### 已完成 ✓

1. **FastSAM语义分割模块** (`semantic_segmentation.py`)
   - [x] FastSAM模型封装
   - [x] 文本提示分割支持
   - [x] 自动分割模式
   - [x] 显存管理（加载/卸载）
   - [x] 语义颜色映射（Cityscapes风格）

2. **语义PLY格式扩展** (`semantic_ply.py`)
   - [x] 扩展PLY头部支持语义标签
   - [x] 语义点云保存函数
   - [x] 水塘采样支持语义标签
   - [x] 语义PLY文件合并

3. **配置文件** (`configs/base_config_semantic_16gb.yaml`)
   - [x] 16GB显存优化配置
   - [x] 语义分割开关
   - [x] FastSAM模型路径配置
   - [x] 语义类别配置

4. **主程序框架** (`da3_semantic_streaming.py`)
   - [x] DA3_Semantic_Streaming类
   - [x] 语义分割集成到process_single_chunk
   - [x] 语义点云保存

### 待完成 ⏳

1. **完善da3_semantic_streaming.py**
   - [ ] 完善process_long_sequence函数
   - [ ] 添加语义融合优化
   - [ ] 实现save_camera_poses函数
   - [ ] 添加进度显示

2. **测试和调试**
   - [ ] 单元测试
   - [ ] 显存使用验证
   - [ ] 端到端测试

3. **文档和使用说明**
   - [ ] 安装指南
   - [ ] 使用教程
   - [ ] 可视化工具说明

---

## 使用方法

### 安装依赖

```bash
# 安装FastSAM
pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git

# 下载FastSAM权重
wget https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.1.0/FastSAM-x.pt
./weights/FastSAM-x.pt
```

### 运行语义分割

```bash
cd da3_streaming

# 基础使用
python da3_semantic_streaming.py \
  --image_dir ./images \
  --config ./configs/base_config_semantic_16gb.yaml \
  --output_dir ./semantic_output

# 恢复模式（从指定块继续）
python da3_semantic_streaming.py \
  --image_dir ./images \
  --config ./configs/base_config_semantic_16gb.yaml \
  --output_dir ./semantic_output \
  --resume_chunk 5
```

### 配置说明

**启用语义分割**:
```yaml
Model:
  semantic_enable: True
```

**指定语义类别** (文本提示模式):
```yaml
Model:
  semantic_classes: ["person", "car", "building", "road", "vegetation", "sky"]
```

**自动模式** (检测COCO 80类):
```yaml
Model:
  semantic_classes: null
```

---

## 输出格式

### PLY文件结构
```
ply
format binary_little_endian 1.0
element vertex <number>
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar semantic_label      # 新增：语义标签ID
property uchar semantic_red        # 新增：语义颜色R
property uchar semantic_green      # 新增：语义颜色G
property uchar semantic_blue       # 新增：语义颜色B
end_header
<点云数据>
```

### 输出文件
```
output_dir/
├── pcd/
│   ├── 0_pcd.ply          # 带语义的点云文件
│   ├── 1_pcd.ply
│   └── combined_pcd.ply   # 合并的完整点云
├── semantics/
│   ├── chunk_0_semantic.npy  # 语义标签数据
│   └── chunk_1_semantic.npy
├── semantic_colormap.npy   # 语义颜色映射表
└── camera_poses.txt         # 相机位姿
```

---

## 性能指标

### 显存使用（理论值）

| 组件 | 显存使用 |
|------|---------|
| DA3推理 | ~3GB |
| FastSAM推理 | ~2.6GB |
| 峰值显存 | ~6GB |
| 安全余量 | ~10GB |

### 处理速度

| 操作 | 速度（估计） |
|------|------------|
| DA3推理 | ~100ms/帧 |
| FastSAM推理 | ~50ms/帧 |
| 总计 | ~150ms/帧 |

---

## 已知问题和解决方案

### 问题1: 显存不足
**解决方案**: 降低chunk_size
```yaml
chunk_size: 32  # 原值48
```

### 问题2: FastSAM模型下载失败
**解决方案**: 手动下载
```bash
# 从GitHub下载
https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.1.0/FastSAM-x.pt
```

### 问题3: 语义类别检测不准确
**解决方案**: 使用文本提示模式指定类别

---

## 下一步工作

1. 完善da3_semantic_streaming.py的实现
2. 添加可视化工具（查看语义点云）
3. 性能优化和测试
4. 编写使用文档

---

**最后更新**: 2026年02月07日 18:17
