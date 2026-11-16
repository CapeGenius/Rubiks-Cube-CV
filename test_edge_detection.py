import numpy as np
import cv2
import edge_detection as cube  # Import
from ultralytics import YOLOE
import yoloe_implementation as yoloe
from pathlib import Path
import kociemba
import sys
from PIL import Image

print(sys.executable)


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


# Additional YOLOE Implementation tests ----------------------------
def test_process_face():
    """
    Check if process face accurately determines the color of each tile on the Rubik's Cube
    """
    model = YOLOE("./runs/segment/train-3/weights/best.pt")
    next_scan_time = 0

    colors_2 = [
        ["Green", "Green", "Red"],
        ["Red", "Red", "Green"],
        ["Red", "Red", "Green"],
    ]
    colors_3 = [
        ["Red", "Green", "Red"],
        ["Green", "Orange", "Orange"],
        ["Green", "Red", "Green"],
    ]
    colors_4 = [
        ["Orange", "Green", "Red"],
        ["Green", "Green", "Green"],
        ["Green", "Orange", "Green"],
    ]

    test_faces = [colors_2, colors_3, colors_4]
    test_output = []

    # use yoloe to segment image

    folder_path = Path(r"./images")

    for item in folder_path.iterdir():
        image = cv2.imread(item)
        results = model.predict(image)
        b_mask_1 = np.zeros(image.shape[:2], np.uint8)
        face = results[0]

        try:
            yoloe.process_face(
                face, b_mask_1, image, next_scan_time, confidence_score=0.8
            )
        except:
            pass

    references = {}

    print(len(yoloe.ordered_scan))

    for face in yoloe.ordered_scan:
        for name, info in yoloe.COLOR_MAPPINGS.items():
            if info["char"] == face:
                references[name] = yoloe.LAB_SCANS[face][4]

    # Classify all piece colors using LAB
    for face in yoloe.ordered_scan:
        colors = []
        for sample in yoloe.LAB_SCANS[face]:
            color = yoloe.classify_lab(sample, references)
            colors.append(color)

        # Make sure the center color is correct
        for name in references:
            if yoloe.COLOR_MAPPINGS[name]["char"] == face:
                colors[4] = name

        # Store colors in 3x3 grid
        yoloe.CENTERS_DICT[face] = [colors[0:3], colors[3:6], colors[6:9]]
        test_output.append([colors[0:3], colors[3:6], colors[6:9]])

    for i in range(len(test_output)):
        print(test_output[i])
        assert test_faces[i] == test_output[i]


def test_double_face():
    """
    Check if the same face scanned twice will not be added again to ordered_scan array
    """
    model = YOLOE("./runs/segment/train-3/weights/best.pt")
    next_scan_time = 0

    # use yoloe to segment image

    folder_path = Path(r"./images")

    i = 0
    # process the same face twice
    for item in folder_path.iterdir():
        i += 1
        image = cv2.imread(item)
        results = model.predict(image)
        b_mask_1 = np.zeros(image.shape[:2], np.uint8)
        face = results[0]

        try:
            yoloe.process_face(
                face, b_mask_1, image, next_scan_time, confidence_score=0.8
            )
            yoloe.process_face(
                face, b_mask_1, image, next_scan_time, confidence_score=0.8
            )
        except:
            pass

    assert len(yoloe.ordered_scan) == i
