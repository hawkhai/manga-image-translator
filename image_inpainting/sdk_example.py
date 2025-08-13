#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manga Inpainting SDK 使用示例

演示如何使用SDK进行图像修复和mask生成
"""

import asyncio
import numpy as np
import cv2
from PIL import Image
import logging
from manga_inpainting_sdk import MangaInpaintingSDK, inpaint_image, generate_mask

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def example_1_direct_inpainting():
    """
    示例1: 直接图像修复
    输入原图和mask，输出修复结果
    """
    print("\n=== 示例1: 直接图像修复 ===")
    
    # 创建SDK实例
    sdk = MangaInpaintingSDK(device='cpu', logger=logger)
    
    # 创建示例图像和mask
    # 这里用代码生成，实际使用时可以从文件加载
    image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    # 创建一个简单的mask（白色区域将被修复）
    mask = np.zeros((400, 600), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (200, 150), 255, -1)  # 矩形区域
    cv2.rectangle(mask, (300, 200), (400, 250), 255, -1)  # 另一个矩形区域
    
    try:
        # 测试不同的修复算法
        algorithms = ['default', 'lama_large', 'none']
        
        for algorithm in algorithms:
            print(f"\n测试算法: {algorithm}")
            result = await sdk.inpaint_image(
                image=image,
                mask=mask,
                algorithm=algorithm,
                inpainting_size=1024
            )
            
            print(f"修复完成 - 输入尺寸: {image.shape}, 输出尺寸: {result.shape}")
            
            # 保存结果（可选）
            # cv2.imwrite(f'result_{algorithm}.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
    except Exception as e:
        print(f"修复失败: {e}")


async def example_2_mask_generation():
    """
    示例2: 从文本框生成mask
    输入原图和文本框坐标，输出mask
    """
    print("\n=== 示例2: 从文本框生成mask ===")
    
    # 创建SDK实例
    sdk = MangaInpaintingSDK(device='cpu', logger=logger)
    
    # 创建示例图像
    image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    # 定义文本框坐标（四个角点）
    textboxes = [
        [[100, 100], [200, 100], [200, 150], [100, 150]],  # 矩形文本框1
        [[300, 200], [450, 200], [450, 280], [300, 280]],  # 矩形文本框2
        [[150, 300], [250, 290], [260, 340], [160, 350]]   # 稍微倾斜的文本框
    ]
    
    try:
        # 测试不同的检测器
        detectors = ['none', 'default']  # 'none'直接基于文本框生成，'default'使用检测器优化
        
        for detector in detectors:
            print(f"\n测试检测器: {detector}")
            mask = await sdk.generate_mask(
                image=image,
                textboxes=textboxes,
                detector=detector,
                detect_size=1024,
                dilation_offset=2  # 膨胀2个像素
            )
            
            print(f"mask生成完成 - 尺寸: {mask.shape}, 非零像素数: {np.count_nonzero(mask)}")
            
            # 保存mask（可选）
            # cv2.imwrite(f'mask_{detector}.jpg', mask)
            
    except Exception as e:
        print(f"mask生成失败: {e}")


async def example_3_one_stop_processing():
    """
    示例3: 一站式处理
    从文本框直接到修复结果
    """
    print("\n=== 示例3: 一站式处理 ===")
    
    # 创建SDK实例
    sdk = MangaInpaintingSDK(device='cpu', logger=logger)
    
    # 创建示例图像
    image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    
    # 定义文本框
    textboxes = [
        [[100, 100], [200, 100], [200, 150], [100, 150]],
        [[300, 200], [400, 200], [400, 250], [300, 250]]
    ]
    
    try:
        # 一次性完成mask生成和图像修复
        result, mask = await sdk.inpaint_with_textboxes(
            image=image,
            textboxes=textboxes,
            inpaint_algorithm='default',
            detector='none',  # 直接基于文本框
            inpainting_size=1024,
            dilation_offset=3
        )
        
        print(f"一站式处理完成")
        print(f"原图尺寸: {image.shape}")
        print(f"mask尺寸: {mask.shape}, 非零像素: {np.count_nonzero(mask)}")
        print(f"结果尺寸: {result.shape}")
        
    except Exception as e:
        print(f"一站式处理失败: {e}")


async def example_4_file_processing():
    """
    示例4: 处理文件
    演示如何处理实际的图像文件
    """
    print("\n=== 示例4: 文件处理示例 ===")
    
    # 创建SDK实例
    sdk = MangaInpaintingSDK(device='cpu', logger=logger)
    
    # 创建示例文件（实际使用时替换为真实文件路径）
    # 这里创建临时文件用于演示
    temp_image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    temp_mask = np.zeros((400, 600), dtype=np.uint8)
    cv2.rectangle(temp_mask, (100, 100), (200, 150), 255, -1)
    
    # 保存临时文件
    cv2.imwrite('temp_image.jpg', cv2.cvtColor(temp_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite('temp_mask.jpg', temp_mask)
    
    try:
        # 从文件路径加载并处理
        result = await sdk.inpaint_image(
            image='temp_image.jpg',  # 文件路径
            mask='temp_mask.jpg',    # 文件路径
            algorithm='default'
        )
        
        print(f"文件处理完成 - 结果尺寸: {result.shape}")
        
        # 保存结果
        cv2.imwrite('result.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print("结果已保存为 result.jpg")
        
    except Exception as e:
        print(f"文件处理失败: {e}")
    
    # 清理临时文件
    import os
    for file in ['temp_image.jpg', 'temp_mask.jpg']:
        if os.path.exists(file):
            os.remove(file)


async def example_5_convenience_functions():
    """
    示例5: 使用便捷函数
    演示如何使用模块级别的便捷函数
    """
    print("\n=== 示例5: 便捷函数 ===")
    
    # 创建示例数据
    image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
    mask = np.zeros((400, 600), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (200, 150), 255, -1)
    
    textboxes = [
        [[300, 200], [400, 200], [400, 250], [300, 250]]
    ]
    
    try:
        # 使用便捷函数进行修复
        result = await inpaint_image(
            image=image,
            mask=mask,
            algorithm='default',
            device='cpu'
        )
        print(f"便捷修复完成 - 结果尺寸: {result.shape}")
        
        # 使用便捷函数生成mask
        generated_mask = await generate_mask(
            image=image,
            textboxes=textboxes,
            detector='none',
            device='cpu'
        )
        print(f"便捷mask生成完成 - mask尺寸: {generated_mask.shape}")
        
    except Exception as e:
        print(f"便捷函数使用失败: {e}")


async def main():
    """主函数：运行所有示例"""
    print("Manga Inpainting SDK 使用示例")
    print("=" * 50)
    
    # 创建SDK实例并检查支持的算法
    sdk = MangaInpaintingSDK()
    supported = sdk.get_supported_algorithms()
    print(f"支持的修复算法: {supported['inpainters']}")
    print(f"支持的检测器: {supported['detectors']}")
    
    # 运行所有示例
    await example_1_direct_inpainting()
    await example_2_mask_generation()
    await example_3_one_stop_processing()
    await example_4_file_processing()
    await example_5_convenience_functions()
    
    print("\n所有示例运行完成！")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
