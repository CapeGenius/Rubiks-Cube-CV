from ultralytics import YOLOE
import cv2
import random
import numpy as np
from skimage.filters import roberts, prewitt

# implemented hough lines based on this opencv article:
#   https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
# implemented live camera with yoloe using this source:
#   https://academy.ultralytics.com/courses/train-your-first-yolo/video-inference


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
            cv2.line(edges, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return edges


def fft_preprocessing(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fft_image = filter_image_fft(gray, 2)
    mag = prewitt(fft_image)
    edges = (mag > 0.05).astype(np.uint8) * 255

    edges = hough_transform(edges)

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


model = YOLOE("yoloe-26m-seg.pt")


# Open the default webcam (0)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    WINDOW = "Cube Detection"
    # edges = fft_preprocessing(frame)
    edges = thresholding_preprocessing(frame)

    cv2.imshow(WINDOW, edges)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


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
