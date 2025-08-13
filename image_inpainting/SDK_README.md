# Manga Inpainting SDK

一个强大的漫画图像修复SDK，提供图像修复和mask生成的完整解决方案。

## 🚀 功能特性

- **🎨 图像修复**: 支持多种先进的修复算法（LAMA Large、LAMA MPE、Stable Diffusion等）
- **🎯 智能Mask生成**: 从文本框坐标自动生成高质量修复mask
- **⚡ 高性能**: 异步处理，支持GPU加速
- **🔧 易于使用**: 简洁的API设计，支持多种输入格式
- **📦 一站式处理**: 从文本框直接到修复结果

## 📋 支持的算法

### 修复算法 (Inpainters)
- **`default`**: AOT算法（推荐，速度快）
- **`lama_large`**: LAMA Large（质量最高）
- **`lama_mpe`**: LAMA MPE（位置感知）
- **`sd`**: Stable Diffusion（生成质量佳）
- **`none`**: 白色填充（调试用）
- **`original`**: 保持原图（跳过修复）

### 检测器 (Detectors)
- **`default`**: 默认检测器
- **`dbconvnext`**: DBNet + ConvNext
- **`ctd`**: 漫画文本检测器
- **`craft`**: CRAFT算法
- **`paddle`**: PaddleOCR检测器
- **`none`**: 直接基于文本框（推荐）

## 🛠 安装要求

确保已安装manga-image-translator项目的所有依赖：

```bash
pip install -r requirements.txt
```

## 📖 使用方法

### 1. 基本图像修复

```python
import asyncio
from manga_inpainting_sdk import MangaInpaintingSDK

async def basic_inpainting():
    # 创建SDK实例
    sdk = MangaInpaintingSDK(device='cpu')  # 或 'cuda' 使用GPU
    
    # 修复图像
    result = await sdk.inpaint_image(
        image='input.jpg',           # 原图路径
        mask='mask.jpg',             # mask图路径
        algorithm='lama_large',      # 修复算法
        inpainting_size=1024         # 处理尺寸
    )
    
    # 保存结果
    import cv2
    cv2.imwrite('result.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

# 运行
asyncio.run(basic_inpainting())
```

### 2. 从文本框生成Mask

```python
async def generate_mask_example():
    sdk = MangaInpaintingSDK(device='cpu')
    
    # 定义文本框坐标（四个角点）
    textboxes = [
        [[100, 100], [200, 100], [200, 150], [100, 150]],  # 矩形框1
        [[300, 200], [400, 200], [400, 250], [300, 250]]   # 矩形框2
    ]
    
    # 生成mask
    mask = await sdk.generate_mask(
        image='input.jpg',
        textboxes=textboxes,
        detector='none',           # 直接基于文本框
        dilation_offset=2          # 膨胀2像素
    )
    
    # 保存mask
    cv2.imwrite('generated_mask.jpg', mask)

asyncio.run(generate_mask_example())
```

### 3. 一站式处理

```python
async def one_stop_processing():
    sdk = MangaInpaintingSDK(device='cpu')
    
    textboxes = [
        [[100, 100], [200, 100], [200, 150], [100, 150]]
    ]
    
    # 从文本框直接到修复结果
    result, mask = await sdk.inpaint_with_textboxes(
        image='input.jpg',
        textboxes=textboxes,
        inpaint_algorithm='lama_large',
        detector='none',
        dilation_offset=3
    )
    
    # 保存结果
    cv2.imwrite('final_result.jpg', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite('used_mask.jpg', mask)

asyncio.run(one_stop_processing())
```

### 4. 便捷函数

```python
from manga_inpainting_sdk import inpaint_image, generate_mask

async def convenience_example():
    # 直接修复
    result = await inpaint_image('input.jpg', 'mask.jpg', 'default')
    
    # 生成mask
    textboxes = [[[100, 100], [200, 100], [200, 150], [100, 150]]]
    mask = await generate_mask('input.jpg', textboxes, 'none')

asyncio.run(convenience_example())
```

## 🔧 API参考

### MangaInpaintingSDK

#### `__init__(device='cpu', logger=None)`
- **device**: 计算设备 (`'cpu'`, `'cuda'`, `'mps'`)
- **logger**: 自定义日志记录器

#### `inpaint_image(image, mask, algorithm='default', inpainting_size=1024, config=None)`
修复图像接口

**参数:**
- **image**: 输入图像（numpy数组、文件路径或PIL图像）
- **mask**: mask图像（白色区域将被修复）
- **algorithm**: 修复算法选择
- **inpainting_size**: 处理时的图像尺寸
- **config**: 修复配置参数

**返回:** `np.ndarray` - 修复后的RGB图像

#### `generate_mask(image, textboxes, detector='default', detect_size=1024, ...)`
生成mask接口

**参数:**
- **image**: 输入图像
- **textboxes**: 文本框坐标列表 `[[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ...]`
- **detector**: 检测器选择
- **detect_size**: 检测时的图像尺寸
- **text_threshold**: 文本阈值 (0.5)
- **box_threshold**: 框阈值 (0.7)
- **unclip_ratio**: 扩展比例 (2.0)
- **dilation_offset**: 膨胀偏移 (0)
- **kernel_size**: 膨胀核大小 (3)

**返回:** `np.ndarray` - 生成的灰度mask（255为需要修复的区域）

#### `inpaint_with_textboxes(image, textboxes, inpaint_algorithm='default', ...)`
一站式处理接口

**返回:** `Tuple[np.ndarray, np.ndarray]` - (修复后图像, 生成的mask)

## 📝 文本框格式说明

文本框使用四个角点坐标定义，格式为：
```python
textboxes = [
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # 文本框1
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],  # 文本框2
    # ... 更多文本框
]
```

- 坐标按顺序排列（通常为左上→右上→右下→左下）
- 支持任意四边形，不限于矩形
- 坐标为整数像素值

## ⚡ 性能优化建议

1. **设备选择**: 使用GPU (`device='cuda'`) 可显著提升处理速度
2. **算法选择**: 
   - 追求速度: `'default'` (AOT)
   - 追求质量: `'lama_large'`
   - 平衡选择: `'default'`
3. **尺寸设置**: 适当降低 `inpainting_size` 和 `detect_size` 可提升速度
4. **批量处理**: 重用SDK实例避免重复加载模型

## 🚨 注意事项

1. **异步处理**: 所有主要方法都是异步的，需要使用 `await` 调用
2. **内存管理**: 处理大图像时注意内存使用，建议适当调整处理尺寸
3. **模型下载**: 首次使用某个算法时会自动下载模型文件
4. **输入格式**: 支持多种输入格式，会自动转换为内部格式
5. **坐标系统**: 使用标准图像坐标系（左上角为原点）

## 🔍 故障排除

### 常见问题

**Q: 修复结果不理想？**
A: 尝试：
- 使用更高质量的算法（如 `lama_large`）
- 调整 `dilation_offset` 参数
- 检查mask质量
- 增加 `inpainting_size`

**Q: 处理速度慢？**
A: 尝试：
- 使用GPU (`device='cuda'`)
- 选择更快的算法 (`'default'`)
- 降低处理尺寸
- 使用 `'none'` 检测器

**Q: 内存不足？**
A: 尝试：
- 降低 `inpainting_size` 和 `detect_size`
- 分批处理大量图像
- 使用 `'none'` 或 `'original'` 算法

## 📚 示例代码

完整的使用示例请参考 `sdk_example.py` 文件，包含：
- 直接图像修复
- Mask生成
- 一站式处理
- 文件处理
- 便捷函数使用

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个SDK！

## 📄 许可证

遵循原项目的许可证条款。
