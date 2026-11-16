import numpy as np
import cv2
import edge_detection as cube  # Import
import yoloe_implementation as yoloe


# Initial Implementation Tests ------------------------------------
def test_nine_colors():
    """Test there are 9 colors per side for each sample"""
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    samples, preview = cube.read_face(image)
    assert len(samples) == 9
    assert preview.shape == image.shape


def test_classify_hsv_color():
    """Check all 6 colors identified correctly"""
    assert cube.classify_hsv_color(0, 255, 255) == "Red"
    assert cube.classify_hsv_color(20, 255, 255) == "Orange"
    assert cube.classify_hsv_color(30, 255, 255) == "Yellow"
    assert cube.classify_hsv_color(60, 255, 255) == "Green"
    assert cube.classify_hsv_color(100, 255, 255) == "Blue"
    assert cube.classify_hsv_color(0, 0, 255) == "White"
    assert cube.classify_hsv_color(145, 255, 255) == "Unknown"


def test_reverse_scramble():
    """Test scramble is reveresed properly"""
    assert cube.invert_moves("R U F'") == "F U' R'"
    assert cube.invert_moves("R2 U' D") == "D' U R2"
    assert cube.invert_moves("R2 D2") == "D2 R2"
    assert cube.invert_moves("") == ""


def test_lab_colors():
    """
    Check if a color is identified correctly against calibrated reference colors
    """
    references = {
        "White": np.array([255, 128, 128]),
        "Red": np.array([100, 150, 150]),
        "Blue": np.array([100, 150, 75]),
    }
    color = np.array([103, 179, 171])
    result = cube.classify_lab(color, references)

    assert result == "Red"


# TODO:
# Edge detection tests?

# Additional YOLOE Implementation tests ----------------------------

# Test we return proper kociemba format
# Test we don't double scan
# Test it can detect wehn a wrong face is scanned
#
