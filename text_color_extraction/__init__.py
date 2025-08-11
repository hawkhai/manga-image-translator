"""
Text Color Extraction SDK

This SDK provides an independent interface for extracting text colors from manga/comic images
using the same OCR models and techniques as the main manga-image-translator project.

Author: Based on manga-image-translator project
License: Apache 2.0
"""

from .color_extractor import TextColorExtractor
from .models import OCRModelWrapper
from .utils import ColorResult, TextRegion

__version__ = "1.0.0"
__all__ = ["TextColorExtractor", "OCRModelWrapper", "ColorResult", "TextRegion"]
