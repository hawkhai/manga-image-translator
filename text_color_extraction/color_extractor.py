"""
Main TextColorExtractor class for the SDK
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Union
from pathlib import Path
import asyncio

from .models import OCRModelWrapper
from .utils import TextRegion, ColorResult, validate_image, adjust_color_contrast


class TextColorExtractor:
    """
    Main class for extracting text colors from manga/comic images
    
    This class provides a high-level interface for text color extraction,
    combining text detection, OCR, and color prediction in one pipeline.
    """
    
    def __init__(self, 
                 model_name: str = "32px",
                 device: str = "cpu"):
        """
        Initialize the text color extractor
        
        Args:
            model_name: OCR model name to use. Options: "32px", "48px", "48px_ctc", "mocr"
            device: Device to run on ("cpu", "cuda", "mps")
        """
        self.model_name = model_name
        self.device = device
        self.ocr_model: Optional[OCRModelWrapper] = None
        self._is_initialized = False
    
    async def initialize(self):
        """Initialize and load the OCR model"""
        if self._is_initialized:
            return
            
        self.ocr_model = OCRModelWrapper(self.model_name)
        await self.ocr_model.load_model(self.device)
        self._is_initialized = True
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available OCR model names"""
        return OCRModelWrapper.get_available_models()
    
    async def extract_from_image(self, 
                               image: Union[str, Path, np.ndarray],
                               regions: Optional[List[TextRegion]] = None,
                               auto_detect: bool = True) -> List[ColorResult]:
        """
        Extract text colors from an image
        
        Args:
            image: Input image (file path or numpy array)
            regions: List of text regions to process (if None, will auto-detect)
            auto_detect: Whether to auto-detect text regions if regions is None
            
        Returns:
            List of ColorResult objects with extracted text and colors
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Load and validate image
        img_array = self._load_image(image)
        if not validate_image(img_array):
            raise ValueError("Invalid image format. Expected RGB numpy array (H, W, 3) with uint8 dtype")
        
        # Get text regions
        if regions is None:
            if auto_detect:
                regions = await self._detect_text_regions(img_array)
            else:
                raise ValueError("Either provide regions or enable auto_detect")
        
        if not regions:
            return []
        
        # Extract text and colors
        updated_regions = await self.ocr_model.extract_text_and_colors(
            img_array, regions
        )
        
        # Convert to ColorResult objects and apply contrast adjustment
        results = []
        for region in updated_regions:
            if region.text.strip() and region.colors:  # Only include regions with text
                fg_adj, bg_adj = adjust_color_contrast(
                    region.colors.foreground, 
                    region.colors.background
                )
                
                result = ColorResult(
                    foreground=fg_adj,
                    background=bg_adj
                )
                results.append(result)
        
        return results
    
    def extract_from_image_sync(self, 
                              image: Union[str, Path, np.ndarray],
                              regions: Optional[List[TextRegion]] = None,
                              auto_detect: bool = True) -> List[ColorResult]:
        """
        Synchronous wrapper for extract_from_image
        
        Args:
            image: Input image (file path or numpy array)
            regions: List of text regions to process
            auto_detect: Whether to auto-detect text regions
            
        Returns:
            List of ColorResult objects
        """
        return asyncio.run(self.extract_from_image(image, regions, auto_detect))
    
    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from file path or validate numpy array"""
        if isinstance(image, (str, Path)):
            # Load from file
            img_bgr = cv2.imread(str(image))
            if img_bgr is None:
                raise ValueError(f"Could not load image from {image}")
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return img_rgb
        elif isinstance(image, np.ndarray):
            return image.copy()
        else:
            raise ValueError("Image must be a file path or numpy array")
    
    async def _detect_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """
        Auto-detect text regions in the image
        
        This is a simplified implementation. For production use,
        you might want to integrate with the detection models from manga_translator.
        """
        # Simple contour-based text detection as fallback
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            # Filter small contours
            area = cv2.contourArea(contour)
            if area < 100:  # Minimum area threshold
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (text regions are usually wider or taller)
            aspect_ratio = w / h
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            region = TextRegion(x=x, y=y, width=w, height=h)
            regions.append(region)
        
        return regions
    
    def create_text_region(self, x: int, y: int, width: int, height: int) -> TextRegion:
        """
        Convenience method to create a TextRegion
        
        Args:
            x: Left coordinate
            y: Top coordinate  
            width: Region width
            height: Region height
            
        Returns:
            TextRegion object
        """
        return TextRegion(x=x, y=y, width=width, height=height)
    
    def create_text_region_from_points(self, points: np.ndarray) -> TextRegion:
        """
        Create TextRegion from quadrilateral points
        
        Args:
            points: Array of 4 points defining the quadrilateral
            
        Returns:
            TextRegion object
        """
        # Get bounding box from points
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        x = int(np.min(x_coords))
        y = int(np.min(y_coords))
        width = int(np.max(x_coords) - x)
        height = int(np.max(y_coords) - y)
        
        return TextRegion(x=x, y=y, width=width, height=height, points=points)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.ocr_model:
            self.ocr_model.unload_model()
        return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.ocr_model is not None:
            self.ocr_model.unload_model()
            self.ocr_model = None
            self._is_initialized = False
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        self.cleanup()
