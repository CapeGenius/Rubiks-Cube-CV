import cv2
import numpy as np

image = cv2.imread("face1_rubiks_cube.jpg")
print("image read")
print(image.shape)

# for edge detections let's just make it grayscale, blur a little bit to reduce noise, and then use Canny
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, threshold1=50, threshold2=150)

# output and display
cv2.imwrite("edges_output_face1.jpg", edges)
cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
