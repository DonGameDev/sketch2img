# tests/test_image_generator.py

import numpy as np
import pytest
from PIL import Image

# Make sure the root of the project is in the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_generator import _classify_shape, analyze_drawing, pil_to_numpy, numpy_to_pil

def test_classify_triangle():
    """Tests if a 3-sided polygon is correctly identified as a triangle."""
    # A simple triangle contour
    triangle_contour = np.array([[[0, 0]], [[10, 0]], [[5, 10]]], dtype=np.int32)
    assert _classify_shape(triangle_contour) == "triangle"

def test_classify_rectangle():
    """Tests if a 4-sided polygon is correctly identified as a rectangle."""
    # A simple rectangle contour
    rect_contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
    assert _classify_shape(rect_contour) == "rectangle"

def test_classify_circle():
    """Tests if a many-sided polygon with high circularity is identified as a circle."""
    # A contour with many vertices approximating a circle
    # This is a simplified heuristic; real circle detection is more complex
    circle_contour = np.array([
        [[10, 0]], [[9, 4]], [[7, 7]], [[4, 9]], [[0, 10]],
        [[-4, 9]], [[-7, 7]], [[-9, 4]], [[-10, 0]], [[-9, -4]],
        [[-7, -7]], [[-4, -9]], [[0, -10]], [[4, -9]], [[7, -7]], [[9, -4]]
    ], dtype=np.int32)
    assert _classify_shape(circle_contour) == "circle"

def test_pil_numpy_conversion():
    """Tests the conversion between PIL Image and NumPy array formats."""
    # Create a simple 10x10 black image
    pil_image = Image.new('RGB', (10, 10), 'black')
    numpy_array = pil_to_numpy(pil_image)

    assert isinstance(numpy_array, np.ndarray)
    assert numpy_array.shape == (10, 10, 3)

    # Test the reverse conversion
    converted_pil = numpy_to_pil(numpy_array)
    assert isinstance(converted_pil, Image.Image)
    assert converted_pil.size == (10, 10)

def test_numpy_to_pil_with_image_editor_dict():
    """Tests conversion when Gradio ImageEditor-style dict data is provided."""
    numpy_array = np.zeros((12, 14, 4), dtype=np.uint8)
    numpy_array[:, :, 3] = 255
    converted_pil = numpy_to_pil({"composite": numpy_array})
    assert isinstance(converted_pil, Image.Image)
    assert converted_pil.mode == "RGB"
    assert converted_pil.size == (14, 12)

def test_numpy_to_pil_accepts_pil_input():
    """Tests conversion when input is already a PIL image."""
    pil_image = Image.new("RGBA", (8, 9), color=(10, 20, 30, 255))
    converted_pil = numpy_to_pil(pil_image)
    assert isinstance(converted_pil, Image.Image)
    assert converted_pil.mode == "RGB"
    assert converted_pil.size == (8, 9)

def test_analyze_drawing_empty_image():
    """Tests that analyzing an empty image returns no shapes or colors."""
    # Create a completely white (empty) image
    white_image = Image.new('RGB', (100, 100), 'white')
    shapes, colors = analyze_drawing(white_image)

    assert len(shapes) == 0
    assert len(colors) == 0
