"""
Test script for the Text Color Extraction SDK

This script provides simple tests to verify that the SDK is working correctly.
"""

import asyncio
import numpy as np
import cv2
import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from text_color_extraction_sdk import TextColorExtractor, TextRegion
    from text_color_extraction_sdk.utils import color_difference, adjust_color_contrast
    print("✓ SDK imports successful")
except ImportError as e:
    print(f"✗ SDK import failed: {e}")
    sys.exit(1)


def create_test_image():
    """Create a test image with colored text regions"""
    # Create a white background image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add some colored rectangles to simulate text backgrounds
    # Region 1: Blue background with black text
    cv2.rectangle(image, (50, 50), (250, 100), (200, 220, 255), -1)  # Light blue
    cv2.putText(image, "Test Text 1", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Region 2: Yellow background with dark text  
    cv2.rectangle(image, (300, 150), (550, 200), (255, 255, 200), -1)  # Light yellow
    cv2.putText(image, "Sample Text", (310, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    # Region 3: Dark background with white text
    cv2.rectangle(image, (100, 250), (400, 300), (50, 50, 50), -1)  # Dark gray
    cv2.putText(image, "White Text", (110, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return image


def test_color_utilities():
    """Test color utility functions"""
    print("\n=== Testing Color Utilities ===")
    
    try:
        # Test color difference
        red = (255, 0, 0)
        green = (0, 255, 0)
        light_red = (255, 100, 100)
        
        diff1 = color_difference(red, green)
        diff2 = color_difference(red, light_red)
        
        print(f"✓ Color difference red-green: {diff1:.2f}")
        print(f"✓ Color difference red-light_red: {diff2:.2f}")
        
        # Test contrast adjustment
        fg = (100, 100, 100)
        bg = (110, 110, 110)  # Very similar to fg
        
        adj_fg, adj_bg = adjust_color_contrast(fg, bg)
        print(f"✓ Contrast adjustment: {fg} + {bg} → {adj_fg} + {adj_bg}")
        
        return True
        
    except Exception as e:
        print(f"✗ Color utilities test failed: {e}")
        return False


def test_text_region():
    """Test TextRegion class"""
    print("\n=== Testing TextRegion ===")
    
    try:
        # Test basic region creation
        region = TextRegion(x=10, y=20, width=100, height=50)
        
        assert region.bbox == (10, 20, 100, 50), "Bbox property failed"
        print("✓ TextRegion creation and bbox property")
        
        # Test quadrilateral points
        points = region.to_quadrilateral_points()
        expected_shape = (4, 2)
        assert points.shape == expected_shape, f"Expected shape {expected_shape}, got {points.shape}"
        print("✓ Quadrilateral points generation")
        
        # Test with custom points
        custom_points = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        region_with_points = TextRegion(x=0, y=0, width=100, height=50, points=custom_points)
        
        retrieved_points = region_with_points.to_quadrilateral_points()
        assert np.array_equal(retrieved_points, custom_points), "Custom points not preserved"
        print("✓ Custom points handling")
        
        return True
        
    except Exception as e:
        print(f"✗ TextRegion test failed: {e}")
        return False


async def test_basic_extraction():
    """Test basic color extraction functionality"""
    print("\n=== Testing Basic Color Extraction ===")
    
    try:
        # Create test image
        test_image = create_test_image()
        print("✓ Test image created")
        
        # Create text regions
        regions = [
            TextRegion(x=50, y=50, width=200, height=50),   # Blue background region
            TextRegion(x=300, y=150, width=250, height=50), # Yellow background region  
            TextRegion(x=100, y=250, width=300, height=50), # Dark background region
        ]
        print(f"✓ Created {len(regions)} test regions")
        
        # Initialize extractor
        extractor = TextColorExtractor(
            model_name="32px",
            device="cpu"  # Use CPU for testing to avoid GPU issues
        )
        
        try:
            # Test extraction
            results = await extractor.extract_from_image(
                image=test_image,
                regions=regions,
                auto_detect=False
            )
            
            print(f"✓ Extraction completed, got {len(results)} results")
            
            # Display results
            for i, result in enumerate(results):
                print(f"  Region {i+1}: '{result.text}' (conf: {result.confidence:.3f})")
                print(f"    FG: {result.foreground_color}, BG: {result.background_color}")
            
            return True
            
        finally:
            pass  # No cleanup needed for new API
            
    except Exception as e:
        print(f"✗ Basic extraction failed: {e}")
        return False


async def test_multiple_ocr_models():
    """Test multiple OCR models"""
    print("Testing multiple OCR models...")
    
    # Get available models
    available_models = TextColorExtractor.get_available_models()
    print(f"Available models: {available_models}")
    
    # Create test image
    test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    test_image[50:100, 50:200] = [50, 50, 50]  # Dark gray region
    
    region = TextRegion(x=50, y=50, width=150, height=50)
    
    success_count = 0
    
    # Test first 2 models to avoid long execution time
    for model_name in available_models[:2]:
        print(f"  Testing {model_name} model...")
        try:
            extractor = TextColorExtractor(model_name=model_name, device="cpu")
            results = await extractor.extract_from_image(
                test_image, 
                regions=[region], 
                auto_detect=False
            )
            print(f"    ✓ {model_name} model worked, found {len(results)} results")
            success_count += 1
            
        except Exception as e:
            print(f"    ✗ {model_name} model failed: {e}")
    
    if success_count > 0:
        print(f"✓ Multiple OCR models test passed ({success_count}/{len(available_models[:2])} models worked)")
        return True
    else:
        print("✗ Multiple OCR models test failed - no models worked")
        return False


async def test_context_manager():
    """Test context manager usage"""
    print("\n=== Testing Context Manager ===")
    
    try:
        test_image = create_test_image()
        regions = [TextRegion(x=100, y=250, width=300, height=50)]
        
        # Test async context manager
        async with TextColorExtractor(model_name="32px", device="cpu") as extractor:
            results = await extractor.extract_from_image(
                image=test_image,
                regions=regions,
                auto_detect=False
            )
            
        print(f"✓ Context manager test completed, got {len(results)} results")
        return True
        
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("Text Color Extraction SDK Test Suite")
    print("=" * 50)
    
    tests = [
        ("Color Utilities", test_color_utilities()),
        ("TextRegion", test_text_region()),
        ("Basic Extraction", test_basic_extraction()),
        ("Multiple OCR Models", test_multiple_ocr_models()),
        ("Context Manager", test_context_manager()),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_coro in tests:
        try:
            if asyncio.iscoroutine(test_coro):
                result = await test_coro
            else:
                result = test_coro
                
            if result:
                passed += 1
                
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    print("Starting SDK tests...")
    
    # Check if we're in the right directory
    if not Path("manga_translator").exists():
        print("Warning: manga_translator directory not found.")
        print("Make sure you're running this from the manga-image-translator root directory.")
    
    # Run tests
    success = asyncio.run(run_all_tests())
    
    if success:
        print("\n✓ SDK is ready for use!")
        sys.exit(0)
    else:
        print("\n✗ SDK tests failed. Please check the setup.")
        sys.exit(1)
