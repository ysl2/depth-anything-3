# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# SegFormer语义分割模块
"""
SegFormer语义分割器

使用SegFormer进行ADE20K 150类语义分割。
特点：
- 支持建筑、墙壁、地面、天空、植被等150类
- 轻量级，显存需求低
- 训练-free
"""

import gc
import os
import numpy as np
import torch
from typing import List, Optional, Dict
import cv2


class SegFormerSegmenter:
    """
    SegFormer语义分割器

    特点：
    - 使用SegFormer进行ADE20K 150类语义分割
    - 支持建筑场景的详细语义类别
    """

    # ADE20K 150类定义 (重要类别)
    ADE20K_CLASSES = {
        0: 'wall', 1: 'building', 2: 'sky', 3: 'floor', 4: 'tree',
        5: 'ceiling', 6: 'road', 7: 'bed', 8: 'window', 9: 'grass',
        10: 'cabinet', 11: 'sidewalk', 12: 'person', 13: 'earth',
        14: 'door', 15: 'table', 16: 'mountain', 17: 'plant',
        18: 'curtain', 19: 'chair', 20: 'car', 21: 'water',
        22: 'painting', 23: 'sofa', 24: 'shelf', 25: 'house',
        26: 'sea', 27: 'rug', 28: 'field', 29: 'armchair',
        30: 'seat', 31: 'fence', 32: 'desk', 33: 'rock',
        34: 'wardrobe', 35: 'lamp', 36: 'bathtub', 37: 'railing',
        38: 'counter', 39: 'sand', 40: 'sink',
        41: 'bookshelf', 42: 'curtain', 43: 'mirror', 44: 'horseshoe',
        45: 'grandstand', 46: 'path', 47: 'floor', 48: 'stairs',
        49: 'runway', 50: 'pool', 51: 'pillow', 52: 'screen door',
        53: 'roof', 54: 'river', 55: 'bridge', 56: 'sand',
        57: 'crate', 58: 'platform', 59: 'field', 60: 'stop sign',
        # ... 更多类别
    }

    # ADE20K颜色映射 (官方颜色)
    ADE20K_COLORS = {
        0: [120, 120, 120],    # wall - 灰色
        1: [180, 120, 120],    # building - 浅红
        2: [6, 230, 230],      # sky - 青色
        3: [80, 50, 50],       # floor - 深褐
        4: [4, 200, 3],        # tree - 绿色
        5: [120, 120, 80],     # ceiling - 浅褐
        6: [140, 140, 140],    # road - 灰色
        7: [204, 5, 255],      # bed - 紫色
        8: [230, 230, 230],    # window - 白色
        9: [4, 250, 7],        # grass - 亮绿
        10: [224, 5, 255],     # cabinet - 紫色
        11: [235, 255, 7],     # sidewalk - 黄色
        12: [150, 5, 61],      # person - 深红
        13: [120, 120, 70],    # earth - 褐色
        14: [80, 180, 20],     # door - 深绿
        15: [220, 220, 220],   # table - 浅灰
        16: [180, 220, 220],   # mountain - 浅青
        17: [20, 60, 255],     # plant - 蓝色
        18: [0, 0, 0],         # curtain - 黑色
        19: [100, 220, 220],   # chair - 青色
        20: [255, 20, 147],    # car - 粉红
        21: [150, 255, 255],   # water - 浅蓝
        22: [255, 255, 255],   # painting - 白色
        23: [200, 200, 200],   # sofa - 浅灰
        24: [255, 255, 0],     # shelf - 黄色
        25: [180, 100, 100],   # house - 浅红
        26: [200, 150, 150],   # sea - 浅红
        27: [150, 100, 100],   # rug - 深红
        28: [100, 150, 100],   # field - 绿色
        29: [150, 150, 100],   # armchair - 黄褐
        30: [100, 100, 150],   # seat - 紫灰
        31: [150, 150, 150],   # fence - 灰色
        32: [100, 150, 150],   # desk - 青灰
        33: [150, 100, 150],   # rock - 紫褐
        34: [100, 100, 100],   # wardrobe - 深灰
        35: [150, 150, 200],   # lamp - 浅蓝
        36: [200, 200, 200],   # bathtub - 浅灰
        37: [150, 150, 150],   # railing - 灰色
        38: [200, 100, 100],   # counter - 红灰
        39: [150, 150, 100],   # sand - 黄褐
        40: [100, 100, 150],   # sink - 紫灰
        41: [100, 150, 100],   # bookshelf - 绿灰
        42: [150, 100, 100],   # curtain - 红褐
        43: [200, 200, 150],   # mirror - 黄灰
        44: [150, 150, 150],   # horseshoe - 灰色
        45: [100, 100, 100],   # grandstand - 深灰
        46: [150, 150, 150],   # path - 灰色
        47: [80, 50, 50],      # floor - 深褐
        48: [120, 120, 120],   # stairs - 灰色
        49: [140, 140, 140],   # runway - 浅灰
        50: [150, 255, 255],   # pool - 浅蓝
        51: [180, 120, 120],   # pillow - 浅红
        52: [150, 150, 150],   # screen door - 灰色
        53: [180, 120, 120],   # roof - 浅红
        54: [150, 255, 255],   # river - 浅蓝
        55: [120, 120, 120],   # bridge - 灰色
        56: [150, 150, 100],   # sand - 黄褐
        57: [150, 150, 150],   # crate - 灰色
        58: [150, 150, 150],   # platform - 灰色
        59: [100, 150, 100],   # field - 绿灰
        60: [200, 200, 0],     # stop sign - 黄色
    }

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b5-finetuned-ade-640-640",
        device: str = "cuda",
    ):
        """
        初始化SegFormer分割器

        Args:
            model_name: HuggingFace模型名称
            device: 设备 ('cuda' 或 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.feature_extractor = None

    def load_model(self):
        """加载SegFormer模型"""
        print(f"[SegFormer] Loading model {self.model_name}...")

        try:
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

            self.feature_extractor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            print(f"[SegFormer] Model loaded successfully")
            return True
        except Exception as e:
            print(f"[SegFormer] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def unload_model(self):
        """卸载模型并释放显存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.feature_extractor is not None:
            del self.feature_extractor
            self.feature_extractor = None
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[SegFormer] Model unloaded, GPU memory freed")

    def segment_images(
        self,
        images: np.ndarray,
    ) -> List[Dict]:
        """
        对图像进行语义分割

        Args:
            images: 输入图像 [N, H, W, 3] numpy数组

        Returns:
            分割结果列表，每个元素包含:
            - 'masks': [H, W] 语义掩码 (类别ID)
            - 'class_colors': [H, W, 3] 类别颜色
        """
        if self.model is None:
            self.load_model()

        results_list = []
        n_images = len(images)

        print(f"[SegFormer] Processing {n_images} images...")

        from PIL import Image
        import torch.nn.functional as F

        for i, img in enumerate(images):
            # 转换为PIL图像
            pil_img = Image.fromarray(img.astype('uint8'))

            # 预处理
            inputs = self.feature_extractor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # [1, num_classes, H, W]

            # 上采样到原始尺寸
            logits = torch.nn.functional.interpolate(
                logits,
                size=img.shape[:2],
                mode="bilinear",
                align_corners=False
            )

            # 获取预测类别
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()  # [H, W]

            # 获取颜色
            semantic_mask = pred.astype(np.uint8)
            class_colors = self._get_color_mask(semantic_mask)

            result = {
                'masks': semantic_mask,
                'class_colors': class_colors
            }
            results_list.append(result)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{n_images} images...")

        return results_list

    def _get_color_mask(self, semantic_mask: np.ndarray) -> np.ndarray:
        """获取语义掩码对应的颜色"""
        H, W = semantic_mask.shape
        color_mask = np.zeros((H, W, 3), dtype=np.uint8)

        for label in np.unique(semantic_mask):
            color = self.ADE20K_COLORS.get(int(label), [128, 128, 128])
            color_mask[semantic_mask == label] = color

        return color_mask

    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        return self.ADE20K_CLASSES.get(class_id, f'class_{class_id}')

    def get_class_color(self, class_id: int) -> List[int]:
        """获取类别颜色"""
        return self.ADE20K_COLORS.get(class_id, [128, 128, 128])

    def print_class_statistics(self, semantic_mask: np.ndarray):
        """打印类别统计"""
        unique, counts = np.unique(semantic_mask, return_counts=True)
        total = semantic_mask.size

        print("\n  语义类别分布:")
        for cls_id, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:20]:
            pct = count / total * 100
            name = self.get_class_name(cls_id)
            color = self.get_class_color(cls_id)
            print(f"    {name:20s} (ID={cls_id:3d}): {count:8,} 像素 ({pct:5.2f}%) - RGB{tuple(color)}")
