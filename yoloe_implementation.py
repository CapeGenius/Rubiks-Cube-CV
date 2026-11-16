from ultralytics import YOLOE
import cv2
import random
import numpy as np
from skimage.filters import roberts, prewitt
import kociemba
import webbrowser
from pathlib import Path
import html
import sys
import time

# used this source for training data: https://docs.ultralytics.com/yolov5/tutorials/train-custom-data
# implemented hough lines based on this opencv article:
#   https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
# implemented live camera with yoloe using this source:
#   https://academy.ultralytics.com/courses/train-your-first-yolo/video-inference
# used this kaggle dataset: https://www.kaggle.com/datasets/aryan7004/rubiks-cube-face-dataset/data
# figured out object isolation from segmentation using this: https://docs.ultralytics.com/guides/isolating-segmentation-objects

CENTERS_DICT = {"B": None, "W": None, "G": None, "R": None, "Y": None, "O": None}
ordered_scan = []
LAB_SCANS = {}


# Define the colors of rubiks cube
COLOR_MAPPINGS = {
    "White": {"char": "W", "id": 1, "bgr": (255, 255, 255)},
    "Yellow": {"char": "Y", "id": 2, "bgr": (0, 255, 255)},
    "Green": {"char": "G", "id": 3, "bgr": (0, 200, 0)},
    "Blue": {"char": "B", "id": 4, "bgr": (255, 0, 0)},
    "Orange": {"char": "O", "id": 5, "bgr": (0, 140, 255)},
    "Red": {"char": "R", "id": 6, "bgr": (0, 0, 255)},
    "Unknown": {"char": "?", "id": 0, "bgr": (128, 128, 128)},
}


def classify_hsv_color(h, s, v):
    """
    Define what the hue value ranges are across the 6 colors of the rubik cube
    """

    # OpenCV Hue ranges from 0 to 180
    if s < 60 and v > 150:
        return "White"
    if 0 <= h <= 10 or 160 <= h <= 180:
        return "Red"  # Red is weird in OpenCv
    elif 11 <= h <= 25:
        return "Orange"
    elif 26 <= h <= 35:
        return "Yellow"
    elif 36 <= h <= 85:
        return "Green"
    elif 86 <= h <= 130:
        return "Blue"
    else:
        return "Unknown"


FACES = "URFDLB"
CENTERS = dict(zip(FACES, ["White", "Red", "Green", "Yellow", "Orange", "Blue"]))
TOP = {"U": "B", "R": "U", "F": "U", "D": "F", "L": "U", "B": "U"}
TARGET_WIDTH = 800  # Downsize to 800px width

# Not using edge detection for now
USE_CONTOURS = False


def face_box(og_image):
    """
    Find bounding box of cube face
    """
    h, w = og_image.shape[:2]

    # Make square to place cube face in
    grid = (w // 6, h // 6, w, h)

    return grid


def classify_lab(color, references):
    """
    Compare each color to calibrated center colors
    """
    closest_color = None
    smallest_distance = float("inf")

    for name in references:
        difference = color - references[name]
        # Give more weight to differences in A & B channels?
        difference = difference * [0.5, 1.5, 1.5]
        distance = np.linalg.norm(difference)  # distance between colors

        # Track closest color
        if distance < smallest_distance:
            smallest_distance = distance
            closest_color = name

    return closest_color


def read_face(og_image):
    """Read colors of 9 squares on cube face"""

    # if face_bounding_box is None:
    #     largest_contour = max(contours, key=cv2.contourArea)
    #     face_bounding_box = cv2.boundingRect(largest_contour)

    fx, fy, fw, fh = face_box(og_image)
    # print(f"Box: X={fx}, Y={fy}, W={fw}, H={fh}")

    # Originally had a complicated way to get squares inside bounding box, now trying ot get the 3x3 and classify colors
    hsv_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2LAB)
    # Divide into 3x3 grid
    step_x = fw / 3.0
    step_y = fh / 3.0
    id_radius = int(min(step_x / 3, step_y / 3))

    square_results = []
    lab_results = []

    annotated_img = og_image.copy()  # to store

    for r in range(3):
        row_colors = []
        for c in range(3):
            cx = int(fx + c * step_x)
            cy = int(fy + (r) * step_y)

            # Finding the central patch and median HSV to hopefully ignore the noise
            patch = hsv_image[
                cy - id_radius : cy + id_radius, cx - id_radius : cx + id_radius
            ]
            med_h = np.median(patch[:, :, 0])
            med_s = np.median(patch[:, :, 1])
            med_v = np.median(patch[:, :, 2])

            color_name = classify_hsv_color(med_h, med_s, med_v)
            row_colors.append(color_name)

            lab_patch = lab_image[
                cy - id_radius : cy + id_radius, cx - id_radius : cx + id_radius
            ]
            lab_results.append(np.median(lab_patch, axis=(0, 1)))

            # Show
            cv2.rectangle(
                annotated_img,
                (cx - id_radius, cy - id_radius),
                (cx + id_radius, cy + id_radius),
                (0, 255, 0),
                1,
            )
            # Make letters more readable on cube
            char = COLOR_MAPPINGS.get(color_name, {"char": "?"})["char"]
            cv2.putText(
                annotated_img,
                char,
                (cx - 9, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                4,
            )
            cv2.putText(
                annotated_img,
                char,
                (cx - 9, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

        square_results.append(row_colors)

    # Draw 3x3 grid over image
    for i in range(4):
        x = int(fx + i * step_x)
        y = int(fy + i * step_y)
        cv2.line(annotated_img, (x, fy), (x, fy + fh), (0, 255, 0), 2)
        cv2.line(annotated_img, (fx, y), (fx + fw, y), (0, 255, 0), 2)

    # Return LAB samples
    return lab_results, annotated_img, square_results


def isolate_image(face, b_mask, frame):
    contour = face.masks.xy[0].astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
    mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
    isolated = cv2.bitwise_and(mask3ch, frame)

    return isolated


def get_results_from_image(face, isolated):
    # Bounding box coordinates
    x1, y1, x2, y2 = face.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
    # Crop the isolated image to the object region
    iso_crop = isolated[y1:y2, x1:x2]
    # cv2.putText(
    #     display,
    #     f"confidence score {face.boxes[0].conf.item()}",
    #     (10, 10),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     (0, 255, 0),
    # )

    return read_face(iso_crop)


def calculate_scan_time(center_char, next_scan_time, lab_results):
    if (
        center_char in CENTERS_DICT
        # and LAB_SCANS.get(center_char) is None
        and time.monotonic() >= next_scan_time
    ):
        # store each face at scan assuming that rotation follows a certain pattern
        LAB_SCANS[center_char] = lab_results
        ordered_scan.append(center_char)
        next_scan_time = time.monotonic() + 5  # Wait 5s before next scan

    return next_scan_time


def process_face(face, b_mask, frame, next_scan_time, confidence_score=0.87):
    if face.boxes[0].conf.item() > confidence_score:
        # print(f"face mask is {face.masks.xy}")

        # gets the image from the mask and creates an isolated image
        isolated = isolate_image(face, b_mask, frame)

        lab_results, annotated_image, square_results = get_results_from_image(
            face, isolated
        )

        # get the center of face
        center_char = COLOR_MAPPINGS[square_results[1][1]]["char"]

        # ASSUME ORIENTATION STEPS: must rotate 2 right, 1 up, then 2 right

        # check if face has been scanned
        next_scan_time = calculate_scan_time(center_char, next_scan_time, lab_results)

        return annotated_image, next_scan_time


def run_yoloe_detection():
    """ """
    model = YOLOE("./runs/segment/train-3/weights/best.pt")
    # model.set_classes(["side-of-rubiks-cube"])

    # Open the default webcam (0)
    cap = cv2.VideoCapture(0)

    WINDOW = "Cube Detection"
    cv2.namedWindow(WINDOW)

    next_scan_time = 0

    # Order to scan faces
    scan_colors = ["Orange", "Green", "Red", "White", "Blue", "Yellow"]

    while cap.isOpened() and len(ordered_scan) < 6:
        success, frame = cap.read()
        if not success:
            break

        # predict based on frame
        results = model.predict(frame)

        for r in results:
            annotated_frame = r.plot()
            # r.masks

        b_mask = np.zeros(frame.shape[:2], np.uint8)
        display = b_mask
        # get the first rubiks cube face

        if len(results) > 0:

            face = results[0]

            if len(face.boxes) > 0:
                # get the box and check confidence score
                if face.boxes[0].conf.item() > 0.87:
                    # print(f"face mask is {face.masks.xy}")

                    annotated_image, next_scan_time = process_face(
                        face, b_mask, frame, next_scan_time
                    )
                    display = annotated_image
                    # cv2.putText(display)

        # Add text to tell when a face has been scanned
        h, w = display.shape[:2]
        i = len(ordered_scan)
        if i < 6:
            message = f"{i}/6  Show {scan_colors[i]} center"
            instruction = "Keep White on top" if i < 3 else "Keep Orange on top"
        else:
            message = "6/6  All faces scanned"
            instruction = "Scanning complete"

        cv2.rectangle(display, (0, 0), (w, 75), (30, 30, 30), -1)
        cv2.putText(
            display,
            message,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display,
            instruction,
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW, display)

        # Press 'q' to exit the video window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Classify colors after all 6 faces are scanned
    if len(ordered_scan) == 6:
        references = {}

        # Get LAB reference colors from centers
        for face in ordered_scan:
            for name, info in COLOR_MAPPINGS.items():
                if info["char"] == face:
                    references[name] = LAB_SCANS[face][4]

        # Classify all piece colors using LAB
        for face in ordered_scan:
            colors = []
            for sample in LAB_SCANS[face]:
                color = classify_lab(sample, references)
                colors.append(color)

            # Make sure the center color is correct
            for name in references:
                if COLOR_MAPPINGS[name]["char"] == face:
                    colors[4] = name

            # Store colors in 3x3 grid
            CENTERS_DICT[face] = [colors[0:3], colors[3:6], colors[6:9]]

            print(face, CENTERS_DICT[face])

    for key, value in CENTERS_DICT.items():
        print(f"{key}: {value}")


def show_twistyplayer(solution, setup=""):
    """
    Update HTML cube viewer with solution and show in browser
    """
    template_path = Path(__file__).resolve().parent / "cube_viewer_template.html"
    output_path = Path(__file__).resolve().parent / "cube_viewer.html"

    page = template_path.read_text(encoding="utf-8")
    page = page.replace("SOLUTION", html.escape(solution, quote=True))
    page = page.replace("SETUP", html.escape(setup, quote=True))
    page = page.replace(
        "Moves: ", "Moves: " + (html.escape(solution) or "Already Solved")
    )

    output_path.write_text(page, encoding="utf-8")
    webbrowser.open(output_path.as_uri())


def invert_moves(moves):
    """
    Reverse scramble to solve the cube
    """
    return " ".join(
        m if m.endswith("2") else m[:-1] if m.endswith("'") else m + "'"
        for m in reversed(moves.split())
    )


def kociemba_solver():
    """
    Order of scan:
        Orange, Green, Red --> w/ White on top
        White, Blue, Yellow --> w/ Orange on top
    """
    # get the center of each face in order --> find mapped scan --> combine into one string
    L_scan = "".join(
        [
            COLOR_MAPPINGS[item]["char"]
            for row in CENTERS_DICT[ordered_scan[0]]
            for item in row
        ]
    )
    F_scan = "".join(
        [
            COLOR_MAPPINGS[item]["char"]
            for row in CENTERS_DICT[ordered_scan[1]]
            for item in row
        ]
    )
    R_scan = "".join(
        [
            COLOR_MAPPINGS[item]["char"]
            for row in CENTERS_DICT[ordered_scan[2]]
            for item in row
        ]
    )
    U_scan = "".join(
        [
            COLOR_MAPPINGS[item]["char"]
            for row in np.rot90(
                CENTERS_DICT[ordered_scan[3]], k=1
            )  # Rotate to work with kociemba
            for item in row
        ]
    )
    B_scan = "".join(
        [
            COLOR_MAPPINGS[item]["char"]
            for row in np.rot90(
                CENTERS_DICT[ordered_scan[4]], k=3
            )  # Rotate to work with kociemba
            for item in row
        ]
    )
    D_scan = "".join(
        [
            COLOR_MAPPINGS[item]["char"]
            for row in np.rot90(
                CENTERS_DICT[ordered_scan[5]], k=1
            )  # Rotate to work with kociemba
            for item in row
        ]
    )

    state = "".join([U_scan, R_scan, F_scan, D_scan, L_scan, B_scan])

    # fix this
    color_map = {
        state[4]: "U",  # R -> U
        state[13]: "R",  # G -> R
        state[22]: "F",  # Y -> F
        state[31]: "D",  # O -> D
        state[40]: "L",  # B -> L
        state[49]: "B",  # W -> B
    }
    cubestring = "".join(color_map[c] for c in state)

    print(f"Final Scan is {cubestring}")
    solution = kociemba.solve(cubestring)
    setup = invert_moves(solution)
    show_twistyplayer(solution, setup)


if __name__ == "__main__":
    run_yoloe_detection()
    print(ordered_scan)
    kociemba_solver()
"""
@misc{kirillov2023segment,
      title={Segment Anything},
      author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
      year={2023},
      eprint={2304.02643},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""


# -------------------------------------- NO LONGER USED --------------------------------
def generate_cicle(original_image, radius_coefficient):
    height, width = original_image.shape[:2]

    center_x = width // 2
    center_y = height // 2
    center = (center_x, center_y)

    radius = min(width, height) // radius_coefficient

    mask = np.zeros((height, width), dtype=np.uint8)
    return cv2.circle(mask, center, radius, color=1, thickness=-1)


def filter_image_fft(original_image, radius_coefficient, blur_filter: bool = False):
    circle_filter = generate_cicle(original_image, radius_coefficient)

    if blur_filter:
        circle_filter = cv2.GaussianBlur(circle_filter, (5, 5), 0)

    f_transform = np.fft.fft2(original_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift))

    filtered_fft_image = f_shift * circle_filter
    unshifted_image = np.fft.ifftshift(filtered_fft_image)
    almost_reconstructed = np.fft.ifft2(unshifted_image)
    reconstructed_image = np.real(almost_reconstructed)

    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)


def hough_transform(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=200,
    minLineLength=80,
    maxLineGap=10,
):

    lines = cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=threshold,
        minLineLength=minLineLength,
        maxLineGap=maxLineGap,
    )

    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            cv2.line(edges, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

    return edges


def fft_preprocessing(
    frame, radius_coef, edge_thresh, hough_thresh, min_line_len, max_line_gap
):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fft_image = filter_image_fft(gray, radius_coef)
    mag = prewitt(fft_image)
    edges = (mag > edge_thresh).astype(np.uint8) * 255

    edges = hough_transform(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    return edges


def thresholding_preprocessing(frame):
    alpha = 1.0
    beta = 1.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    median = cv2.medianBlur(gray, 5)
    dst = cv2.adaptiveThreshold(
        median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3
    )

    # edges = cv2.Canny(dst, 20, 200)

    edges = hough_transform(dst)
    return edges


def nothing(_):
    pass
