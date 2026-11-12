import cv2
import numpy as np
import sys
import html
import webbrowser
from pathlib import Path
import kociemba

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

# # The original image size is (4032, 3024, 3) which is too large
# # Load raw image
# raw_image = cv2.imread("face1_rubiks_cube.jpg")
# if raw_image is None:
#     print("Could not find image")

# # Downsize to 800px width
# TARGET_WIDTH = 800
# h_orig, w_orig = raw_image.shape[:2]
# scale = TARGET_WIDTH / float(w_orig)
# target_height = int(h_orig * scale)
# og_image = cv2.resize(
#     raw_image, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA
# )
# img_area = og_image.shape[0] * og_image.shape[1]
# # og_image = cv2.imread("face1_rubiks_cube.jpg")
# # print("image read")
# # print(og_image.shape)


def face_box(og_image):
    """
    Find bounding box of cube face
    """
    h, w = og_image.shape[:2]

    # Make square to place cube face in
    size = min(int(min(h, w) * 0.6), h - 190)
    size = max(60, size)
    grid = ((w - size) // 2, (h - size) // 2, size, size)

    if not USE_CONTOURS:
        return grid

    img_area = h * w
    # For edge detections - using grayscale, blur a little bit to reduce noise,
    # and then use Canny edge detection
    gray = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, threshold1=30, threshold2=100)

    # # output and display - visual check
    # cv2.imwrite("edges_output_face1.jpg", edges)
    # # cv2.imshow("Canny Edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Close gap for the edges that are not defined the full way
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # full_edges = cv2.dilate(closed_edges, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(
        closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Now easiest to find the 'bounding box' to help in ignoring outside noise
    # contours, hierarchy = cv2.findContours(
    #     full_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    # )
    # print(f"Total contours: {len(contours)}")  # check how much it's detecting

    face_bounding_box = None
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        a = cv2.contourArea(
            c
        )  # calculates the area of a detected contour using Green's theorem
        if (
            0.05 * img_area < a < 0.85 * img_area
        ):  # this logic may be off...double check
            x, y, w, h = cv2.boundingRect(c)
            if 0.65 <= (w / float(h)) <= 1.35:
                face_bounding_box = (x, y, w, h)
                break

    return face_bounding_box or grid


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
    id_radius = int(
        min(step_x, step_y) * 0.15
    )  # where it should look for the squares of rubiks cubs with 15% buffer

    square_results = []
    lab_results = []

    annotated_img = og_image.copy()  # to store

    # 3 rows each with 3 squares
    for r in range(3):
        row_colors = []
        for c in range(3):
            # Finding the center points
            cx = int(fx + (c + 0.5) * step_x)
            cy = int(fy + (r + 0.5) * step_y)

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
    return lab_results, annotated_img


# # Print the layour for double check
# print("\nDetected Face Colors (3x3):")
# for row in square_results:
#     print(" ".join(f"[{c:^7}]" for c in row))

# # create image
# cv2.imwrite("detected_face1.jpg", annotated_img)

# OLD CODE

# # Now need to extract the squares inside the bounding box
# expected_sq_area = (fw * fh) / 9
# cube_squares = []


def classify_lab(color, references):
    """
    Compare each color to calibrated center colors
    """
    closest_color = None
    smallest_distance = float("inf")

    for name in references:
        difference = color - references[name]
        # Give more weight to differences in A & B channels
        difference = difference * [0.5, 1.5, 1.5]
        distance = np.linalg.norm(difference) # distance between colors

        # Track closest color
        if distance < smallest_distance:
            smallest_distance = distance
            closest_color = name

    return closest_color


def scan_cube():
    """
    Scan cube with webcam
    """
    camera = cv2.VideoCapture(0)
    scans, references = {}, {}
    i = 0

    #
    while i < 6:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Camera disconnected")

        # Resize
        h_orig, w_orig = frame.shape[:2]
        if w_orig > TARGET_WIDTH:
            frame = cv2.resize(
                frame,
                (TARGET_WIDTH, int(h_orig * TARGET_WIDTH / w_orig)),
                interpolation=cv2.INTER_AREA,
            )

        samples, preview = read_face(frame)
        face = FACES[i]
        h, w = preview.shape[:2]

        # Face instructions
        cv2.rectangle(preview, (0, 0), (w, 75), (30, 30, 30), -1)
        cv2.putText(
            preview,
            f"{i+1}/6  {face} - {CENTERS[face]} center",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            preview,
            f"Keep {TOP[face]} ({CENTERS[TOP[face]]}) on top",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Keyboard Controls for UI
        cv2.rectangle(preview, (0, h - 40), (w, h), (30, 30, 30), -1)
        cv2.putText(
            preview,
            "SPACE save    B back    Q quit",
            (12, h - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Cube Scanner", preview)
        key = cv2.waitKey(1) & 255

        # Quit key
        if key in (27, ord("q")):
            camera.release()
            cv2.destroyAllWindows()
            return None
        # Go back to prev face
        if key == ord("b") and i > 0:
            i -= 1
        # Save current face
        elif key == 32:
            # Learn actual color of faces center automatically
            scans[face] = samples
            references[CENTERS[face]] = samples[4]
            print(f"Saved {face} / {CENTERS[face]} center:", samples[4].astype(int))
            i += 1

    camera.release()
    cv2.destroyAllWindows()

    # Since all 6 reference colors exist --> reclassify each saved sticker
    color_to_face = {color: face for face, color in CENTERS.items()}
    state = ""
    for face in FACES:
        colors = [classify_lab(sample, references) for sample in scans[face]]

        # Print detected colors
        print(face, " ".join(COLOR_MAPPINGS[color]["char"] for color in colors))

        # Convert color names into cube notation for kociemba
        state += "".join(color_to_face[color] for color in colors)

    # TODO: Add auto color count adjustment
    if any(state.count(face) != 9 for face in FACES):
        raise ValueError("Wrong color counts --> rescan")
    return state


def invert_moves(moves):
    """
    Reverse scramble to solve the cube
    """
    return " ".join(
        m if m.endswith("2") else m[:-1] if m.endswith("'") else m + "'"
        for m in reversed(moves.split())
    )


def show_twistyplayer(solution, setup=""):
    """
    Update HTML cube viewer with solution and show in browser
    """
    path = Path(__file__).resolve().parent / "cube_viewer.html"

    page = path.read_text(encoding="utf-8")
    page = page.replace("SOLUTION", html.escape(solution, quote=True))
    page = page.replace("SETUP", html.escape(setup, quote=True))
    page = page.replace(
        "Moves: ", "Moves: " + (html.escape(solution) or "Already Solved")
    )

    path.write_text(page, encoding="utf-8")
    webbrowser.open(path.as_uri())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        solution = " ".join(sys.argv[1:]).replace("’", "'")
        setup = ""
    else:
        state = scan_cube()
        # Stop if quit during scanning
        if state is None:
            sys.exit()

        solved = "".join(face * 9 for face in FACES)

        solution = "" if state == solved else kociemba.solve(state)
        setup = invert_moves(solution)
        print("Solution:", solution or "Already solved")

    show_twistyplayer(solution, setup)
