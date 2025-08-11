"""
Utility classes and functions for text color extraction SDK
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ColorResult:
    """Result of color extraction for a text region"""
    text: str
    confidence: float
    foreground_color: Tuple[int, int, int]  # RGB values (0-255)
    background_color: Tuple[int, int, int]  # RGB values (0-255)
    bbox: Tuple[int, int, int, int]  # x, y, width, height


@dataclass
class TextRegion:
    """Represents a text region with its bounding box"""
    x: int
    y: int
    width: int
    height: int
    points: Optional[np.ndarray] = None  # For quadrilateral regions
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
    
    def to_quadrilateral_points(self) -> np.ndarray:
        """Convert bbox to quadrilateral points"""
        if self.points is not None:
            return self.points
        
        # Convert bbox to quadrilateral points (top-left, top-right, bottom-right, bottom-left)
        x, y, w, h = self.x, self.y, self.width, self.height
        return np.array([
            [x, y],         # top-left
            [x + w, y],     # top-right
            [x + w, y + h], # bottom-right
            [x, y + h]      # bottom-left
        ], dtype=np.float32)


def color_difference(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    """
    Calculate color difference using CIE76 formula
    
    Args:
        rgb1: First RGB color tuple
        rgb2: Second RGB color tuple
        
    Returns:
        Color difference value
    """
    import cv2
    
    color1 = np.array(rgb1, dtype=np.uint8).reshape(1, 1, 3)
    color2 = np.array(rgb2, dtype=np.uint8).reshape(1, 1, 3)
    
    # Convert to LAB color space for better perceptual difference
    lab1 = cv2.cvtColor(color1, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(color2, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    diff = lab1 - lab2
    diff[..., 0] *= 0.392  # Weight L channel
    
    return np.linalg.norm(diff, axis=2).item()


def adjust_color_contrast(fg_color: Tuple[int, int, int], 
                         bg_color: Tuple[int, int, int],
                         min_difference: float = 30.0) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Adjust background color if contrast with foreground is too low
    
    Args:
        fg_color: Foreground RGB color
        bg_color: Background RGB color  
        min_difference: Minimum color difference threshold
        
    Returns:
        Tuple of (adjusted_fg_color, adjusted_bg_color)
    """
    if color_difference(fg_color, bg_color) < min_difference:
        fg_avg = np.mean(fg_color)
        # Choose white or black background based on foreground brightness
        adjusted_bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
        return fg_color, adjusted_bg
    
    return fg_color, bg_color


def validate_image(image: np.ndarray) -> bool:
    """
    Validate input image format
    
    Args:
        image: Input image array
        
    Returns:
        True if image is valid, False otherwise
    """
    if not isinstance(image, np.ndarray):
        return False
    
    if len(image.shape) != 3:
        return False
        
    if image.shape[2] != 3:
        return False
        
    if image.dtype != np.uint8:
        return False
        
    return True
