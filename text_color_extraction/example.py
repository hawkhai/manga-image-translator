"""
Example usage of the Text Color Extraction SDK

This script demonstrates how to use the SDK to extract text colors from manga/comic images
with support for multiple OCR models.
"""

import asyncio
import numpy as np
from pathlib import Path

from text_color_extraction_sdk import TextColorExtractor, TextRegion, ColorResult


async def basic_usage_example():
    """Basic usage example with auto-detection"""
    print("=== Basic Usage Example ===")
    
    # Initialize extractor with default 32px model
    extractor = TextColorExtractor(model_name="32px", device="cpu")
    
    # Create a sample image (in practice, you'd load a real image)
    sample_image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
    
    # Add some colored text regions (simulated)
    sample_image[50:100, 50:200] = [0, 0, 0]      # Black text area
    sample_image[150:200, 50:200] = [255, 0, 0]   # Red text area
    sample_image[250:300, 50:200] = [0, 255, 0]   # Green text area
    
    try:
        # Extract colors with auto-detection
        results = await extractor.extract_from_image(sample_image, auto_detect=True)
        
        print(f"Found {len(results)} text regions:")
        for i, result in enumerate(results):
            print(f"  Region {i+1}:")
            print(f"    Foreground: RGB{result.foreground}")
            print(f"    Background: RGB{result.background}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")


async def multiple_models_example():
    """Example demonstrating different OCR models"""
    print("=== Multiple OCR Models Example ===")
    
    # Show available models
    available_models = TextColorExtractor.get_available_models()
    print(f"Available OCR models: {available_models}")
    
    # Create sample image
    sample_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    sample_image[100:150, 100:300] = [50, 50, 50]  # Dark gray text area
    
    # Define a specific region to analyze
    regions = [TextRegion(x=100, y=100, width=200, height=50)]
    
    # Test different models
    for model_name in available_models[:2]:  # Test first 2 models to avoid long execution
        print(f"\n--- Testing {model_name} model ---")
        try:
            extractor = TextColorExtractor(model_name=model_name, device="cpu")
            results = await extractor.extract_from_image(sample_image, regions=regions)
            
            if results:
                result = results[0]
                print(f"  Foreground: RGB{result.foreground}")
                print(f"  Background: RGB{result.background}")
            else:
                print("  No text detected")
                
        except Exception as e:
            print(f"  Error with {model_name}: {e}")


async def manual_regions_example():
    """Example with manually specified regions"""
    print("=== Manual Regions Example ===")
    
    extractor = TextColorExtractor(model_name="32px", device="cpu")
    
    # Create sample image
    sample_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Define specific regions to analyze
    regions = [
        TextRegion(x=50, y=50, width=150, height=50),   # Top region
        TextRegion(x=50, y=150, width=150, height=50),  # Middle region
        TextRegion(x=50, y=250, width=150, height=50),  # Bottom region
    ]
    
    try:
        results = await extractor.extract_from_image(sample_image, regions=regions)
        
        print(f"Analyzed {len(regions)} specified regions:")
        for i, result in enumerate(results):
            print(f"  Region {i+1}: FG{result.foreground} BG{result.background}")
            
    except Exception as e:
        print(f"Error: {e}")


def sync_example():
    """Example using synchronous API"""
    print("=== Synchronous API Example ===")
    
    extractor = TextColorExtractor(model_name="32px", device="cpu")
    
    # Create sample image
    sample_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    try:
        # Use synchronous method
        results = extractor.extract_from_image_sync(sample_image, auto_detect=True)
        print(f"Synchronously found {len(results)} regions")
        
    except Exception as e:
        print(f"Error: {e}")


async def context_manager_example():
    """Example using context manager"""
    print("=== Context Manager Example ===")
    
    sample_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    async with TextColorExtractor(model_name="32px", device="cpu") as extractor:
        results = await extractor.extract_from_image(sample_image, auto_detect=True)
        print(f"Context manager found {len(results)} regions")


def color_utilities_example():
    """Example of color utility functions"""
    print("=== Color Utilities Example ===")
    
    from text_color_extraction_sdk.utils import calculate_color_difference, adjust_color_contrast
    
    # Test color difference calculation
    color1 = (255, 0, 0)    # Red
    color2 = (0, 255, 0)    # Green
    color3 = (255, 100, 100)  # Light red
    
    diff1 = calculate_color_difference(color1, color2)
    diff2 = calculate_color_difference(color1, color3)
    
    print(f"Color difference between red and green: {diff1:.2f}")
    print(f"Color difference between red and light red: {diff2:.2f}")
    
    # Test contrast adjustment
    fg_color = (100, 100, 100)  # Gray
    bg_color = (120, 120, 120)  # Slightly lighter gray
    
    adjusted_fg, adjusted_bg = adjust_color_contrast(fg_color, bg_color)
    print(f"Original: FG{fg_color} BG{bg_color}")
    print(f"Adjusted: FG{adjusted_fg} BG{adjusted_bg}")


async def device_selection_example():
    """Example showing device selection"""
    print("=== Device Selection Example ===")
    
    sample_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Try different devices (CPU is most compatible)
    devices = ["cpu"]  # Add "cuda" or "mps" if available
    
    for device in devices:
        print(f"\n--- Testing on {device.upper()} ---")
        try:
            extractor = TextColorExtractor(model_name="32px", device=device)
            results = await extractor.extract_from_image(sample_image, auto_detect=True)
            print(f"  Successfully processed on {device}")
            
        except Exception as e:
            print(f"  Error on {device}: {e}")


async def main():
    """Run all examples"""
    print("Text Color Extraction SDK Examples")
    print("=" * 50)
    
    await basic_usage_example()
    await multiple_models_example()
    await manual_regions_example()
    sync_example()
    await context_manager_example()
    color_utilities_example()
    await device_selection_example()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())
