import cv2
import numpy as np

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


# The original image size is (4032, 3024, 3) which is too large
# Load raw image
raw_image = cv2.imread("face1_rubiks_cube.jpg")
if raw_image is None:
    print("Could not find image")

# Downsize to 800px width
TARGET_WIDTH = 800
h_orig, w_orig = raw_image.shape[:2]
scale = TARGET_WIDTH / float(w_orig)
target_height = int(h_orig * scale)
og_image = cv2.resize(
    raw_image, (TARGET_WIDTH, target_height), interpolation=cv2.INTER_AREA
)
img_area = og_image.shape[0] * og_image.shape[1]
# og_image = cv2.imread("face1_rubiks_cube.jpg")
# print("image read")
# print(og_image.shape)

# For edge detections - using grayscale, blur a little bit to reduce noise, and then use Canny edge detection
gray = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, threshold1=30, threshold2=100)

# output and display - visual check
cv2.imwrite("edges_output_face1.jpg", edges)
# cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

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
print(f"Total contours: {len(contours)}")  # check how much it's detecting

face_bounding_box = None
for c in sorted(contours, key=cv2.contourArea, reverse=True):
    a = cv2.contourArea(
        c
    )  # calculates the area of a detected contour using Green's theorem
    if 0.05 * img_area < a < 0.85 * img_area:  # this logic may be off...double check
        x, y, w, h = cv2.boundingRect(c)
        if 0.65 <= (w / float(h)) <= 1.35:
            face_bounding_box = (x, y, w, h)
            break

if face_bounding_box is None:
    face_bounding_box = max(contours, key=cv2.contourArea)

fx, fy, fw, fh = face_bounding_box
print(f"Box: X={fx}, Y={fy}, W={fw}, H={fh}")


# Originally had a complicated way to get squares inside bounding box, now trying ot get the 3x3 and classify colors
hsv_image = cv2.cvtColor(og_image, cv2.COLOR_BGR2HSV)
step_x = fw / 3.0
step_y = fh / 3.0
id_radius = int(
    min(step_x, step_y) * 0.15
)  # where it should look for the squares of rubiks cubs with 15% buffer

square_results = []
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

        # Show
        cv2.rectangle(
            annotated_img,
            (cy - id_radius, cy + id_radius),
            (cx - id_radius, cx + id_radius),
            (0, 255, 0),
            1,
        )
        char = COLOR_MAPPINGS.get(color_name, {"char": "?"})["char"]
        cv2.putText(
            annotated_img,
            char,
            (cx - 7, cy + 7),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            (0, 0, 0),
            2,
        )
    square_results.append(row_colors)

# Print the layour for double check
print("\nDetected Face Colors (3x3):")
for row in square_results:
    print(" ".join(f"[{c:^7}]" for c in row))

# create image
cv2.imwrite("detected_face1.jpg", annotated_img)


# OLD CODE

# # Now need to extract the squares inside the bounding box
# expected_sq_area = (fw * fh) / 9
# cube_squares = []
