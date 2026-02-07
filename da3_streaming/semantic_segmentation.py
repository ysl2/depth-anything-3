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
FastSAM语义分割模块

集成FastSAM用于DA3-Streaming的语义分割功能。
特点：
- 轻量级：~68M参数，显存需求~2.6GB
- 快速：比SAM快50倍
- Training-free：直接使用预训练模型
- 支持文本提示
"""

import gc
import os
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
import cv2


class FastSAMSegmenter:
    """
    FastSAM语义分割器

    特点：
    - 模型大小仅~68MB
    - 推理显存~2.6GB
    - 比SAM快50倍
    - 支持文本提示
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

    # 自定义语义类别映射
    SEMANTIC_MAPPING = {
        # 室内场景
        'floor': 0, 'wall': 1, 'ceiling': 2, 'table': 3, 'chair': 4,
        'bed': 5, 'sofa': 6, 'couch': 6, 'window': 7, 'door': 8,
        'desk': 9, 'shelf': 10, 'cabinet': 11,
        # 室外场景
        'road': 20, 'street': 20, 'sidewalk': 21, 'ground': 22,
        'building': 23, 'house': 23, 'wall': 1,
        'vegetation': 24, 'tree': 24, 'plant': 24, 'grass': 25,
        'sky': 26, 'cloud': 26,
        'vehicle': 27, 'car': 27, 'truck': 28, 'bus': 29,
        'person': 30, 'people': 30, 'human': 30,
        # 物体
        'object': 99, 'thing': 99,
    }

    def __init__(
        self,
        model_path: str = "./weights/FastSAM-x.pt",
        device: str = "cuda",
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.9,
        retina_masks: bool = True,
        imgsz: int = 1024,
    ):
        """
        初始化FastSAM分割器

        Args:
            model_path: FastSAM模型权重路径
            device: 设备 ('cuda' 或 'cpu')
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            retina_masks: 是否使用精细掩码
            imgsz: 输入图像尺寸
        """
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.retina_masks = retina_masks
        self.imgsz = imgsz

        # 初始化模型
        self.model = None
        self.model_loaded = False

        # 预定义类别到ID的映射
        self.class_to_id = self._build_class_to_id_mapping()

    def _build_class_to_id_mapping(self) -> Dict[str, int]:
        """构建类别名称到ID的映射"""
        mapping = {}
        # 添加COCO类别
        for i, cls_name in enumerate(self.COCO_CLASSES):
            mapping[cls_name] = i

        # 添加自定义语义类别
        for cls_name, cls_id in self.SEMANTIC_MAPPING.items():
            mapping[cls_name] = cls_id

        return mapping

    def load_model(self):
        """加载FastSAM模型（延迟加载以节省显存）"""
        if self.model_loaded:
            return

        print(f"[FastSAM] Loading model from {self.model_path}...")

        try:
            # 尝试导入FastSAM
            from fastsam import FastSAM, FastSAMPrompt

            self.model = FastSAM(self.model_path)
            self.model.to(self.device)
            self.model_loaded = True
            self.FastSAMPrompt = FastSAMPrompt

            print(f"[FastSAM] Model loaded successfully on {self.device}")

        except ImportError:
            raise ImportError(
                "FastSAM not installed. Install with:\n"
                "  pip install git+https://github.com/CASIA-IVA-Lab/FastSAM.git"
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
        images: List[str] | np.ndarray,
        text_prompts: Optional[List[str]] = None,
        return_masks: bool = True,
    ) -> Dict:
        """
        对图像列表进行语义分割

        Args:
            images: 图像路径列表或numpy数组 (N, H, W, 3)
            text_prompts: 文本提示列表，如 ["person", "car", "building"]
            return_masks: 是否返回掩码

        Returns:
            results: 包含以下键的字典
                - masks: (N, H, W) 语义标签图
                - boxes: 检测框列表
                - classes: 类别ID列表
        """
        # 确保模型已加载
        self.load_model()

        num_images = len(images)
        print(f"[FastSAM] Processing {num_images} images...")

        # 初始化结果
        all_semantic_masks = []
        all_boxes = []
        all_classes = []

        for i, img in enumerate(images):
            # 加载图像
            if isinstance(img, str):
                image_np = cv2.imread(img)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            elif isinstance(img, np.ndarray):
                if img.ndim == 3:
                    image_np = img
                else:
                    raise ValueError(f"Unsupported image shape: {img.shape}")
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            H, W = image_np.shape[:2]

            # 运行FastSAM推理
            results = self.model(
                image_np,
                device=self.device,
                retina_masks=self.retina_masks,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
            )

            # 处理结果
            prompt_process = self.FastSAMPrompt(image_np, results, device=self.device)

            # 初始化语义掩码（背景=0）
            semantic_mask = np.zeros((H, W), dtype=np.uint8)

            if text_prompts is not None and len(text_prompts) > 0:
                # 使用文本提示模式
                for class_id, prompt in enumerate(text_prompts):
                    # 标准化提示词
                    prompt_normalized = prompt.lower().strip()

                    # 获取掩码
                    masks = prompt_process.text_prompt(text=prompt)

                    if masks is not None and len(masks) > 0:
                        for mask in masks:
                            if mask is not None:
                                mask_np = mask if isinstance(mask, np.ndarray) else mask.cpu().numpy()
                                # 调整掩码尺寸到原始图像
                                if mask_np.shape != (H, W):
                                    mask_np = cv2.resize(mask_np.astype(np.uint8), (W, H),
                                                       interpolation=cv2.INTER_NEAREST)
                                semantic_mask[mask_np > 0] = class_id + 1  # 背景为0
            else:
                # 自动模式：使用所有检测到的对象
                if results[0].masks is not None and len(results[0].masks) > 0:
                    masks = results[0].masks
                    boxes = results[0].boxes
                    classes = results[0].boxes.cls.cpu().numpy() if boxes is not None else []

                    for j, mask in enumerate(masks):
                        if mask is not None:
                            mask_np = mask if isinstance(mask, np.ndarray) else mask.cpu().numpy()
                            if mask_np.shape != (H, W):
                                mask_np = cv2.resize(mask_np.astype(np.uint8), (W, H),
                                                   interpolation=cv2.INTER_NEAREST)

                            class_id = int(classes[j]) + 1 if j < len(classes) else 1
                            semantic_mask[mask_np > 0] = class_id

                    # 保存检测框信息
                    all_boxes.append(boxes)
                    all_classes.append(classes)

            all_semantic_masks.append(semantic_mask)

            if (i + 1) % 10 == 0:
                print(f"[FastSAM] Processed {i+1}/{num_images} images")

        # 转换为numpy数组
        all_semantic_masks = np.array(all_semantic_masks, dtype=np.uint8)

        return {
            'masks': all_semantic_masks,
            'boxes': all_boxes,
            'classes': all_classes,
        }

    def get_memory_usage(self) -> Dict[str, float]:
        """获取当前显存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
            }
        return {'allocated_gb': 0, 'reserved_gb': 0}

    def get_semantic_colormap(self) -> np.ndarray:
        """
        获取语义颜色映射表 (Cityscapes风格)

        Returns:
            colormap: (num_classes, 3) RGB颜色
        """
        # Cityscapes风格颜色
        colors = [
            [128, 64, 128],   # road
            [244, 35, 232],   # sidewalk
            [70, 70, 70],     # building
            [102, 102, 156],  # wall
            [190, 153, 153],  # fence
            [153, 153, 153],  # pole
            [250, 170, 30],   # traffic light
            [220, 220, 0],    # traffic sign
            [107, 142, 35],   # vegetation
            [152, 251, 152],  # terrain
            [70, 130, 180],   # sky
            [220, 20, 60],    # person
            [255, 0, 0],      # rider
            [0, 0, 142],      # car
            [0, 0, 70],       # truck
            [0, 60, 100],     # bus
            [0, 80, 100],     # train
            [0, 0, 230],      # motorcycle
            [119, 11, 32],    # bicycle
        ]

        # 扩展到更多类别
        for i in range(len(colors), 100):
            # 随机颜色
            color = np.random.randint(0, 255, 3).tolist()
            colors.append(color)

        return np.array(colors, dtype=np.uint8)


def segment_with_memory_cleanup(
    segmenter: FastSAMSegmenter,
    images: List[str] | np.ndarray,
    text_prompts: Optional[List[str]] = None,
) -> Dict:
    """
    使用显存清理的语义分割

    Args:
        segmenter: FastSAM分割器实例
        images: 图像列表
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
        # 无论成功与否，都释放显存
        segmenter.unload_model()

    return results


def download_fastsam_weights(output_dir: str = "./weights") -> str:
    """
    下载FastSAM预训练权重

    Args:
        output_dir: 输出目录

    Returns:
        权重文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # FastSAM-x权重URL
    weights_url = "https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.1.0/FastSAM-x.pt"
    weights_path = os.path.join(output_dir, "FastSAM-x.pt")

    if os.path.exists(weights_path):
        print(f"[FastSAM] Weights already exist: {weights_path}")
        return weights_path

    print(f"[FastSAM] Downloading weights from {weights_url}...")

    try:
        import urllib.request
        urllib.request.urlretrieve(weights_url, weights_path)
        print(f"[FastSAM] Weights downloaded to: {weights_path}")
        return weights_path
    except Exception as e:
        print(f"[FastSAM] Failed to download weights: {e}")
        print("[FastSAM] Please download manually from:")
        print("  https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v0.1.0/FastSAM-x.pt")
        raise
