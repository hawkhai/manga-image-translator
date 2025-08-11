"""
OCR Model wrapper for text color extraction SDK
"""

import os
import sys
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Type
from pathlib import Path

# Add the parent directory to the path to import from manga_translator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manga_translator.ocr.model_32px import Model32pxOCR
from manga_translator.ocr.model_48px import Model48pxOCR
from manga_translator.ocr.model_48px_ctc import Model48pxCTCOCR
from manga_translator.ocr.model_manga_ocr import ModelMangaOCR
from manga_translator.ocr.common import OfflineOCR
from manga_translator.config import Ocr, OcrConfig
from manga_translator.utils import Quadrilateral
from .utils import TextRegion, ColorResult


# Registry of available OCR models
OCR_MODELS: Dict[str, Type[OfflineOCR]] = {
    "32px": Model32pxOCR,
    "48px": Model48pxOCR, 
    "48px_ctc": Model48pxCTCOCR,
    "mocr": ModelMangaOCR,
}


class OCRModelWrapper:
    """
    Wrapper class for OCR models that handles model loading and inference
    for text and color extraction.
    """
    
    def __init__(self, model_name: str = "32px"):
        """
        Initialize the OCR model wrapper.
        
        Args:
            model_name: Name of the OCR model to use. 
                       Options: "32px", "48px", "48px_ctc", "mocr"
        """
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        
        if model_name not in OCR_MODELS:
            available = ", ".join(OCR_MODELS.keys())
            raise ValueError(f"Unsupported model '{model_name}'. Available models: {available}")
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available OCR model names."""
        return list(OCR_MODELS.keys())
    
    async def load_model(self, device: str = "cpu") -> None:
        """
        Load the OCR model asynchronously.
        
        Args:
            device: Device to load the model on ("cpu", "cuda", etc.)
        """
        if self.is_loaded:
            return
            
        try:
            model_class = OCR_MODELS[self.model_name]
            self.model = model_class()
            await self.model.download()
            await self.model.load(device)
            self.is_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load OCR model '{self.model_name}': {e}")
    
    async def extract_text_and_colors(
        self, 
        image: np.ndarray, 
        regions: List[TextRegion],
        config: Optional[OcrConfig] = None,
        verbose: bool = False
    ) -> List[TextRegion]:
        """
        Extract text and colors from specified regions in the image.
        
        Args:
            image: Input image as numpy array (RGB format)
            regions: List of text regions to process
            config: OCR configuration (optional)
            verbose: Whether to print verbose output
            
        Returns:
            List of TextRegion objects with extracted text and colors
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert TextRegion objects to Quadrilateral objects
        quadrilaterals = []
        for region in regions:
            # Use the to_quadrilateral_points method to get proper numpy array format
            points = region.to_quadrilateral_points()
            quad = Quadrilateral(points, text="", prob=1.0)
            quadrilaterals.append(quad)
        
        try:
            # Use default config if none provided
            if config is None:
                config = OcrConfig()
            
            # Run OCR inference
            results = await self.model.recognize(image, quadrilaterals, config, verbose)
            
            # Convert results back to TextRegion objects
            updated_regions = []
            for quad in results:
                # Extract colors (default to white/black if not available)
                fg_color = (255, 255, 255)  # Default white
                bg_color = (0, 0, 0)        # Default black
                
                if hasattr(quad, 'fg_r') and hasattr(quad, 'fg_g') and hasattr(quad, 'fg_b'):
                    fg_color = (int(quad.fg_r), int(quad.fg_g), int(quad.fg_b))
                
                if hasattr(quad, 'bg_r') and hasattr(quad, 'bg_g') and hasattr(quad, 'bg_b'):
                    bg_color = (int(quad.bg_r), int(quad.bg_g), int(quad.bg_b))
                
                region = TextRegion(
                    bbox=quad.pts,
                    text=quad.text,
                    confidence=quad.prob,  # Use prob instead of confidence
                    colors=ColorResult(
                        text=quad.text,
                        confidence=quad.prob,
                        foreground_color=fg_color,
                        background_color=bg_color,
                        bbox=(0, 0, 0, 0)  # Will be updated from quad.pts if needed
                    )
                )
                updated_regions.append(region)
            
            return updated_regions
            
        except Exception as e:
            raise RuntimeError(f"OCR inference failed: {e}")
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self.model = None
        self.is_loaded = False


# Legacy function removed - use OCRModelWrapper directly
