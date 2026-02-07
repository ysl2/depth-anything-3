# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# 语义分割模块 - 使用YOLOv8进行COCO 80类检测
"""
YOLOv8语义分割器

使用YOLOv8进行COCO 80类物体检测和语义分割。
特点：
- 检测80类COCO物体
- 提供精确的类别标签
- 训练-free
"""

import gc
import os
import numpy as np
import torch
from typing import List, Optional, Dict, Tuple
import cv2


class YOLOSegmenter:
    """
    YOLOv8语义分割器

    特点：
    - 使用YOLOv8进行80类COCO物体检测
    - 为每个检测框分配语义标签
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
        'person': [220, 20, 60],
        'bicycle': [119, 11, 32],
        'car': [0, 0, 142],
        'motorcycle': [0, 0, 230],
        'airplane': [0, 60, 100],
        'bus': [0, 80, 100],
        'train': [0, 0, 230],
        'truck': [0, 0, 70],
        'boat': [0, 0, 142],
        'traffic light': [250, 170, 30],
        'fire hydrant': [0, 0, 0],
        'stop sign': [220, 220, 0],
        'bench': [107, 142, 35],
        'bird': [70, 130, 180],
        'cat': [220, 20, 60],
        'dog': [220, 20, 60],
        'horse': [220, 20, 60],
        'sheep': [220, 20, 60],
        'cow': [220, 20, 60],
        'potted plant': [107, 142, 35],
        'couch': [152, 251, 152],
        'chair': [70, 130, 180],
        'bed': [128, 64, 128],
        'dining table': [70, 70, 70],
        'toilet': [152, 251, 152],
        'tv': [70, 70, 70],
        'laptop': [70, 70, 70],
        'mouse': [70, 70, 70],
        'remote': [0, 0, 142],
        'keyboard': [70, 70, 70],
        'cell phone': [0, 0, 142],
        'microwave': [128, 64, 128],
        'oven': [128, 64, 128],
        'toaster': [0, 0, 142],
        'sink': [70, 130, 180],
        'refrigerator': [128, 64, 128],
        'book': [70, 130, 180],
        'clock': [70, 70, 70],
        'vase': [0, 0, 0],
        'background': [70, 70, 70],
    }

    def __init__(
        self,
        model_name: str = "yolov8x.pt",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
    ):
        """
        初始化YOLOv8分割器

        Args:
            model_name: YOLOv8模型名称 (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            device: 设备 ('cuda' 或 'cpu')
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            imgsz: 输入图像尺寸
        """
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.model = None
        self.model_path = None

        # 构建类别到颜色的映射
        self._build_color_map()

    def _build_color_map(self):
        """构建类别ID到颜色的映射"""
        self.class_colors = {}
        for cls_id, cls_name in enumerate(self.COCO_CLASSES):
            if cls_name in self.SEMANTIC_COLORS:
                self.class_colors[cls_id] = self.SEMANTIC_COLORS[cls_name]
            else:
                # 为未定义的类别分配随机颜色
                np.random.seed(cls_id)
                self.class_colors[cls_id] = list(np.random.randint(50, 200, 3).tolist())

    def load_model(self):
        """加载YOLOv8模型"""
        print(f"[YOLOv8] Loading model {self.model_name}...")

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            self.model_path = self.model_name
            print(f"[YOLOv8] Model loaded successfully")
            return True
        except Exception as e:
            print(f"[YOLOv8] Error loading model: {e}")
            return False

    def unload_model(self):
        """卸载模型并释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[YOLOv8] Model unloaded, GPU memory freed")

    def segment_images(
        self,
        images: np.ndarray,
        text_prompts: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        对图像进行语义分割

        Args:
            images: 输入图像 [N, H, W, 3] numpy数组
            text_prompts: 可选的文本提示（用于过滤检测类别）

        Returns:
            分割结果列表，每个元素包含:
            - 'masks': [H, W] 语义掩码 (类别ID)
            - 'boxes': 检测框列表
            - 'classes': 检测到的类别名称列表
            - 'confidences': 置信度列表
        """
        if self.model is None:
            self.load_model()

        results_list = []
        n_images = len(images)

        print(f"[YOLOv8] Processing {n_images} images...")

        for i, img in enumerate(images):
            # 运行YOLOv8检测
            results = self.model(
                img,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            # 创建语义掩码
            H, W = img.shape[:2]
            semantic_mask = np.zeros((H, W), dtype=np.uint8)

            # 收集检测结果
            boxes = []
            classes = []
            confidences = []

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes_obj = results[0].boxes
                cls_ids = boxes_obj.cls.cpu().numpy() if hasattr(boxes_obj, 'cls') else None
                xyxy = boxes_obj.xyxy.cpu().numpy() if hasattr(boxes_obj, 'xyxy') else None
                confs = boxes_obj.conf.cpu().numpy() if hasattr(boxes_obj, 'conf') else None

                if cls_ids is not None and xyxy is not None:
                    for j, (cls_id, box) in enumerate(zip(cls_ids, xyxy)):
                        cls_id = int(cls_id)
                        cls_name = self.COCO_CLASSES[cls_id] if cls_id < len(self.COCO_CLASSES) else f'class_{cls_id}'

                        # 过滤类别（如果提供了文本提示）
                        if text_prompts is not None and cls_name not in text_prompts:
                            continue

                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.astype(int)

                        # 限制在图像范围内
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W, x2), min(H, y2)

                        # 标记该区域为对应的类别ID (使用+1以保留0为背景)
                        semantic_mask[y1:y2, x1:x2] = cls_id + 1

                        boxes.append([x1, y1, x2, y2])
                        classes.append(cls_name)
                        if confs is not None and j < len(confs):
                            confidences.append(float(confs[j]))

            result = {
                'masks': semantic_mask,
                'boxes': boxes,
                'classes': classes,
                'confidences': confidences
            }
            results_list.append(result)

        return results_list

    def get_class_color(self, class_name: str) -> List[int]:
        """获取类别的颜色"""
        if class_name in self.SEMANTIC_COLORS:
            return self.SEMANTIC_COLORS[class_name]
        elif class_name == 'background':
            return [70, 70, 70]
        else:
            return [128, 128, 128]

    def get_class_id_color(self, class_id: int) -> List[int]:
        """获取类别ID对应的颜色"""
        if class_id in self.class_colors:
            return self.class_colors[class_id]
        elif class_id == -1:  # background
            return [70, 70, 70]
        else:
            return [128, 128, 128]
