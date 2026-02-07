# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
语义分割模块 - 使用Ultralytics FastSAM

使用Ultralytics的原生FastSAM实现进行语义分割。
特点：
- 轻量级：~68M参数
- 快速：基于YOLOv8
- Training-free
"""

import gc
import os
import numpy as np
import torch
from typing import List, Optional, Dict
import cv2


class FastSAMSegmenter:
    """
    FastSAM语义分割器 (Ultralytics版本)

    特点：
    - 使用Ultralytics原生FastSAM
    - 模型大小~68MB
    - 推理显存~2.6GB
    """

    # COCO 80类定义
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # 语义颜色映射 (Cityscapes风格)
    SEMANTIC_COLORS = {
        'road': [128, 64, 128],
        'sidewalk': [244, 35, 232],
        'building': [70, 70, 70],
        'wall': [102, 102, 156],
        'fence': [190, 153, 153],
        'pole': [153, 153, 153],
        'traffic light': [250, 170, 30],
        'traffic sign': [220, 220, 0],
        'vegetation': [107, 142, 35],
        'terrain': [152, 251, 152],
        'sky': [70, 130, 180],
        'person': [220, 20, 60],
        'rider': [255, 0, 0],
        'car': [0, 0, 142],
        'truck': [0, 0, 70],
        'bus': [0, 60, 100],
        'train': [0, 80, 100],
        'motorcycle': [0, 0, 230],
        'bicycle': [119, 11, 32],
    }

    def __init__(
        self,
        model_path: str = "FastSAM-x.pt",
        device: str = "cuda",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.9,
        imgsz: int = 1024,
    ):
        """
        初始化FastSAM分割器

        Args:
            model_path: FastSAM模型权重路径
            device: 设备 ('cuda' 或 'cpu')
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            imgsz: 输入图像尺寸
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        # 初始化模型
        self.model = None
        self.model_loaded = False

        # 构建类别到ID的映射
        self.class_to_id = {cls: i for i, cls in enumerate(self.COCO_CLASSES)}

    def load_model(self):
        """加载FastSAM模型"""
        if self.model_loaded:
            return

        print(f"[FastSAM] Loading model from {self.model_path}...")

        try:
            from ultralytics import FastSAM

            self.model = FastSAM(self.model_path)
            self.model.to(self.device)
            self.model_loaded = True
            print("[FastSAM] Model loaded successfully")

        except ImportError:
            raise ImportError(
                "Ultralytics not installed. Install with:\n"
                "  pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load FastSAM model: {e}")

    def unload_model(self):
        """卸载模型释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False
            torch.cuda.empty_cache()
            gc.collect()
            print("[FastSAM] Model unloaded, GPU memory freed")

    def segment_images(
        self,
        images: np.ndarray,
        text_prompts: Optional[List[str]] = None,
    ) -> Dict:
        """
        对图像进行语义分割

        Args:
            images: (N, H, W, 3) 图像数组
            text_prompts: 文本提示列表 (可选)

        Returns:
            results: 包含语义掩码的字典
        """
        # 确保模型已加载
        self.load_model()

        N, H, W = images.shape[:3]
        all_semantic_masks = []

        print(f"[FastSAM] Processing {N} images...")

        for i, img in enumerate(images):
            # 运行推理
            results = self.model(
                img,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
            )

            # 初始化语义掩码（背景=0）
            semantic_mask = np.zeros((H, W), dtype=np.uint8)

            # 处理检测结果
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                masks = results[0].masks

                if masks is not None:
                    for j, mask in enumerate(masks):
                        if mask is not None:
                            mask_np = mask.data.cpu().numpy()
                            if mask_np.ndim == 3:
                                mask_np = mask_np[0]

                            # 调整掩码尺寸
                            if mask_np.shape != (H, W):
                                mask_np = cv2.resize(
                                    mask_np.astype(np.uint8),
                                    (W, H),
                                    interpolation=cv2.INTER_NEAREST
                                )

                            # 获取类别ID
                            if hasattr(boxes, 'cls') and boxes.cls is not None:
                                class_id = int(boxes.cls[j]) + 1
                            else:
                                class_id = 1  # 默认类别

                            # 合并掩码
                            semantic_mask[mask_np > 0] = class_id

            all_semantic_masks.append(semantic_mask)

        all_semantic_masks = np.array(all_semantic_masks, dtype=np.uint8)

        return {
            'masks': all_semantic_masks,
        }

    def get_semantic_colormap(self) -> np.ndarray:
        """
        获取语义颜色映射表

        Returns:
            colormap: (num_classes, 3) RGB颜色
        """
        # Cityscapes风格颜色
        colors = [
            [0, 0, 0],         # 背景
            [128, 64, 128],    # road
            [244, 35, 232],    # sidewalk
            [70, 70, 70],      # building
            [102, 102, 156],   # wall
            [190, 153, 153],   # fence
            [153, 153, 153],   # pole
            [250, 170, 30],    # traffic light
            [220, 220, 0],     # traffic sign
            [107, 142, 35],    # vegetation
            [152, 251, 152],   # terrain
            [70, 130, 180],    # sky
            [220, 20, 60],     # person
            [255, 0, 0],       # rider
            [0, 0, 142],       # car
            [0, 0, 70],        # truck
            [0, 60, 100],      # bus
            [0, 80, 100],      # train
            [0, 0, 230],       # motorcycle
            [119, 11, 32],     # bicycle
        ]

        # 扩展到更多类别
        for i in range(len(colors), 100):
            color = np.random.randint(0, 255, 3).tolist()
            colors.append(color)

        return np.array(colors, dtype=np.uint8)


def segment_with_memory_cleanup(
    segmenter: FastSAMSegmenter,
    images: np.ndarray,
    text_prompts: Optional[List[str]] = None,
) -> Dict:
    """
    使用显存清理的语义分割

    Args:
        segmenter: FastSAM分割器实例
        images: 图像数组
        text_prompts: 文本提示

    Returns:
        分割结果字典
    """
    # 加载模型
    segmenter.load_model()

    try:
        # 执行分割
        results = segmenter.segment_images(images, text_prompts)
    finally:
        # 释放显存
        segmenter.unload_model()

    return results
