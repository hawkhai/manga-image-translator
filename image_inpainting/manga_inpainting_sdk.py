#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manga Image Inpainting SDK

提供两个核心接口：
1. inpaint_image: 修复图片接口（原图+mask+算法选择→修复结果）
2. generate_mask: mask生成接口（原图+文本框+算法选择→mask图片）

Author: Manga Image Translator Team
"""

import asyncio
import numpy as np
import cv2
from typing import List, Tuple, Union, Optional
from PIL import Image
import logging

# 导入项目模块
from manga_translator.inpainting import get_inpainter, prepare as prepare_inpainter, dispatch as dispatch_inpainter
from manga_translator.detection import get_detector, prepare as prepare_detector, dispatch as dispatch_detector
from manga_translator.mask_refinement import complete_mask
from manga_translator.config import Inpainter, Detector, InpainterConfig
from manga_translator.utils import Quadrilateral, TextBlock


class MangaInpaintingSDK:
    """
    漫画图像修复SDK
    
    提供图像修复和mask生成的完整解决方案
    """
    
    def __init__(self, device: str = 'cpu', logger: Optional[logging.Logger] = None):
        """
        初始化SDK
        
        Args:
            device: 计算设备 ('cpu', 'cuda', 'mps')
            logger: 日志记录器
        """
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        # 支持的算法映射
        self.inpainter_mapping = {
            'default': Inpainter.default,
            'lama_large': Inpainter.lama_large,
            'lama_mpe': Inpainter.lama_mpe,
            'sd': Inpainter.sd,
            'none': Inpainter.none,
            'original': Inpainter.original,
        }
        
        self.detector_mapping = {
            'default': Detector.default,
            'dbconvnext': Detector.dbconvnext,
            'ctd': Detector.ctd,
            'craft': Detector.craft,
            'paddle': Detector.paddle,
            'none': Detector.none,
        }
        
        # 缓存已加载的模型
        self._loaded_inpainters = set()
        self._loaded_detectors = set()
    
    def _validate_image(self, image: Union[np.ndarray, str, Image.Image]) -> np.ndarray:
        """
        验证和转换输入图像
        
        Args:
            image: 输入图像（numpy数组、文件路径或PIL图像）
            
        Returns:
            np.ndarray: RGB格式的numpy数组
        """
        if isinstance(image, str):
            # 从文件路径加载
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法加载图像文件: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            # PIL图像转换
            img = np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if len(img.shape) == 3 and img.shape[2] == 3:
                # 假设已经是RGB格式
                pass
            elif len(img.shape) == 3 and img.shape[2] == 4:
                # RGBA转RGB
                img = img[:, :, :3]
            else:
                raise ValueError("不支持的图像格式")
        else:
            raise ValueError("不支持的图像类型")
        
        return img.astype(np.uint8)
    
    def _validate_mask(self, mask: Union[np.ndarray, str, Image.Image]) -> np.ndarray:
        """
        验证和转换输入mask
        
        Args:
            mask: 输入mask（numpy数组、文件路径或PIL图像）
            
        Returns:
            np.ndarray: 灰度格式的numpy数组
        """
        if isinstance(mask, str):
            # 从文件路径加载
            mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
            if mask_img is None:
                raise ValueError(f"无法加载mask文件: {mask}")
        elif isinstance(mask, Image.Image):
            # PIL图像转换
            mask_img = np.array(mask.convert('L'))
        elif isinstance(mask, np.ndarray):
            if len(mask.shape) == 3:
                # 多通道转灰度
                mask_img = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            else:
                mask_img = mask.copy()
        else:
            raise ValueError("不支持的mask类型")
        
        return mask_img.astype(np.uint8)
    
    def _textboxes_to_quadrilaterals(self, textboxes: List[List[List[int]]]) -> List[Quadrilateral]:
        """
        将文本框坐标转换为Quadrilateral对象
        
        Args:
            textboxes: 文本框列表，每个文本框为4个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            List[Quadrilateral]: Quadrilateral对象列表
        """
        quadrilaterals = []
        for i, box in enumerate(textboxes):
            if len(box) != 4 or not all(len(point) == 2 for point in box):
                self.logger.warning(f"跳过无效的文本框 {i}: {box}")
                continue
            
            points = np.array(box, dtype=np.float32)
            quad = Quadrilateral(points, text=f"text_{i}", prob=1.0)
            quadrilaterals.append(quad)
        
        return quadrilaterals
    
    async def inpaint_image(
        self,
        image: Union[np.ndarray, str, Image.Image],
        mask: Union[np.ndarray, str, Image.Image],
        algorithm: str = 'default',
        inpainting_size: int = 1024,
        config: Optional[InpainterConfig] = None
    ) -> np.ndarray:
        """
        修复图片接口
        
        Args:
            image: 原始图像
            mask: mask图像（白色区域将被修复）
            algorithm: 修复算法选择 ('default', 'lama_large', 'lama_mpe', 'sd', 'none', 'original')
            inpainting_size: 修复时的图像尺寸
            config: 修复配置参数
            
        Returns:
            np.ndarray: 修复后的图像 (RGB格式)
            
        Raises:
            ValueError: 输入参数无效
            RuntimeError: 修复过程出错
        """
        try:
            # 验证输入
            img = self._validate_image(image)
            mask_img = self._validate_mask(mask)
            
            if algorithm not in self.inpainter_mapping:
                raise ValueError(f"不支持的修复算法: {algorithm}. 支持的算法: {list(self.inpainter_mapping.keys())}")
            
            # 检查图像和mask尺寸匹配
            if img.shape[:2] != mask_img.shape[:2]:
                self.logger.warning("图像和mask尺寸不匹配，将调整mask尺寸")
                mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]))
            
            # 获取修复器
            inpainter_key = self.inpainter_mapping[algorithm]
            
            # 准备模型（如果尚未加载）
            if inpainter_key not in self._loaded_inpainters:
                self.logger.info(f"正在准备修复器: {algorithm}")
                await prepare_inpainter(inpainter_key, self.device)
                self._loaded_inpainters.add(inpainter_key)
            
            # 执行修复
            self.logger.info(f"开始修复图像，算法: {algorithm}")
            result = await dispatch_inpainter(
                inpainter_key, img, mask_img, config, inpainting_size, self.device, verbose=True
            )
            
            self.logger.info("图像修复完成")
            return result
            
        except Exception as e:
            self.logger.error(f"图像修复失败: {str(e)}")
            raise RuntimeError(f"图像修复失败: {str(e)}")
    
    async def generate_mask(
        self,
        image: Union[np.ndarray, str, Image.Image],
        textboxes: List[List[List[int]]],
        detector: str = 'default',
        detect_size: int = 1024,
        text_threshold: float = 0.5,
        box_threshold: float = 0.7,
        unclip_ratio: float = 2.0,
        dilation_offset: int = 0,
        kernel_size: int = 3
    ) -> np.ndarray:
        """
        mask生成接口
        
        Args:
            image: 原始图像
            textboxes: 文本框坐标列表，格式为 [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]
            detector: 检测器选择 ('default', 'dbconvnext', 'ctd', 'craft', 'paddle', 'none')
            detect_size: 检测时的图像尺寸
            text_threshold: 文本阈值
            box_threshold: 框阈值
            unclip_ratio: 扩展比例
            dilation_offset: 膨胀偏移
            kernel_size: 核大小
            
        Returns:
            np.ndarray: 生成的mask图像 (灰度格式，255为需要修复的区域)
            
        Raises:
            ValueError: 输入参数无效
            RuntimeError: mask生成过程出错
        """
        try:
            # 验证输入
            img = self._validate_image(image)
            
            if not textboxes:
                self.logger.warning("未提供文本框，返回空mask")
                return np.zeros(img.shape[:2], dtype=np.uint8)
            
            if detector not in self.detector_mapping:
                raise ValueError(f"不支持的检测器: {detector}. 支持的检测器: {list(self.detector_mapping.keys())}")
            
            # 转换文本框格式
            quadrilaterals = self._textboxes_to_quadrilaterals(textboxes)
            if not quadrilaterals:
                self.logger.warning("没有有效的文本框，返回空mask")
                return np.zeros(img.shape[:2], dtype=np.uint8)
            
            self.logger.info(f"处理 {len(quadrilaterals)} 个文本框")
            
            # 如果选择none检测器，直接基于文本框生成mask
            if detector == 'none':
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                for quad in quadrilaterals:
                    # 填充四边形区域
                    pts = quad.pts.astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                
                # 应用膨胀
                if dilation_offset > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    mask = cv2.dilate(mask, kernel, iterations=dilation_offset)
                
                return mask
            
            # 使用检测器生成初始mask
            detector_key = self.detector_mapping[detector]
            
            # 准备检测器（如果尚未加载）
            if detector_key not in self._loaded_detectors:
                self.logger.info(f"正在准备检测器: {detector}")
                await prepare_detector(detector_key)
                self._loaded_detectors.add(detector_key)
            
            # 执行检测获取初始mask
            self.logger.info(f"使用检测器 {detector} 生成初始mask")
            textlines, raw_mask, mask = await dispatch_detector(
                detector_key, img, detect_size, text_threshold, box_threshold, unclip_ratio,
                invert=False, gamma_correct=False, rotate=False, auto_rotate=False,
                device=self.device, verbose=True
            )
            
            # 如果检测器返回了mask，使用complete_mask进行优化
            if mask is not None and len(quadrilaterals) > 0:
                self.logger.info("优化mask质量")
                try:
                    optimized_mask = complete_mask(
                        img, mask, quadrilaterals, 
                        keep_threshold=1e-2, 
                        dilation_offset=dilation_offset,
                        kernel_size=kernel_size
                    )
                    if optimized_mask is not None:
                        mask = optimized_mask
                except Exception as e:
                    self.logger.warning(f"mask优化失败，使用原始mask: {str(e)}")
            
            # 如果没有检测到内容，基于输入的文本框生成mask
            if mask is None or mask.sum() == 0:
                self.logger.info("检测器未找到内容，基于输入文本框生成mask")
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                for quad in quadrilaterals:
                    pts = quad.pts.astype(np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                
                # 应用膨胀
                if dilation_offset > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    mask = cv2.dilate(mask, kernel, iterations=dilation_offset)
            
            self.logger.info("mask生成完成")
            return mask
            
        except Exception as e:
            self.logger.error(f"mask生成失败: {str(e)}")
            raise RuntimeError(f"mask生成失败: {str(e)}")
    
    async def inpaint_with_textboxes(
        self,
        image: Union[np.ndarray, str, Image.Image],
        textboxes: List[List[List[int]]],
        inpaint_algorithm: str = 'default',
        detector: str = 'default',
        inpainting_size: int = 1024,
        detect_size: int = 1024,
        dilation_offset: int = 0,
        config: Optional[InpainterConfig] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        一站式修复接口：从文本框直接到修复结果
        
        Args:
            image: 原始图像
            textboxes: 文本框坐标列表
            inpaint_algorithm: 修复算法
            detector: 检测器选择
            inpainting_size: 修复尺寸
            detect_size: 检测尺寸
            dilation_offset: 膨胀偏移
            config: 修复配置
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (修复后图像, 生成的mask)
        """
        # 生成mask
        mask = await self.generate_mask(
            image, textboxes, detector, detect_size,
            dilation_offset=dilation_offset
        )
        
        # 执行修复
        result = await self.inpaint_image(
            image, mask, inpaint_algorithm, inpainting_size, config
        )
        
        return result, mask
    
    def get_supported_algorithms(self) -> dict:
        """
        获取支持的算法列表
        
        Returns:
            dict: 包含修复算法和检测器的字典
        """
        return {
            'inpainters': list(self.inpainter_mapping.keys()),
            'detectors': list(self.detector_mapping.keys())
        }


# 便捷函数
async def inpaint_image(
    image: Union[np.ndarray, str, Image.Image],
    mask: Union[np.ndarray, str, Image.Image],
    algorithm: str = 'default',
    device: str = 'cpu',
    inpainting_size: int = 1024
) -> np.ndarray:
    """
    便捷的图像修复函数
    
    Args:
        image: 原始图像
        mask: mask图像
        algorithm: 修复算法
        device: 计算设备
        inpainting_size: 修复尺寸
        
    Returns:
        np.ndarray: 修复后的图像
    """
    sdk = MangaInpaintingSDK(device=device)
    return await sdk.inpaint_image(image, mask, algorithm, inpainting_size)


async def generate_mask(
    image: Union[np.ndarray, str, Image.Image],
    textboxes: List[List[List[int]]],
    detector: str = 'default',
    device: str = 'cpu',
    detect_size: int = 1024
) -> np.ndarray:
    """
    便捷的mask生成函数
    
    Args:
        image: 原始图像
        textboxes: 文本框坐标列表
        detector: 检测器选择
        device: 计算设备
        detect_size: 检测尺寸
        
    Returns:
        np.ndarray: 生成的mask
    """
    sdk = MangaInpaintingSDK(device=device)
    return await sdk.generate_mask(image, textboxes, detector, detect_size)


if __name__ == "__main__":
    # 示例用法
    async def main():
        # 创建SDK实例
        sdk = MangaInpaintingSDK(device='cpu')
        
        # 示例1: 直接修复图像
        # result = await sdk.inpaint_image('input.jpg', 'mask.jpg', 'lama_large')
        
        # 示例2: 从文本框生成mask
        textboxes = [
            [[100, 100], [200, 100], [200, 150], [100, 150]],  # 矩形文本框
            [[300, 200], [400, 200], [400, 250], [300, 250]]   # 另一个文本框
        ]
        # mask = await sdk.generate_mask('input.jpg', textboxes, 'default')
        
        # 示例3: 一站式处理
        # result, mask = await sdk.inpaint_with_textboxes('input.jpg', textboxes, 'lama_large')
        
        print("SDK初始化完成")
        print("支持的算法:", sdk.get_supported_algorithms())
    
    # 运行示例
    asyncio.run(main())
