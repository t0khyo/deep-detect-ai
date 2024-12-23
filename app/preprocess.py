import cv2
import numpy as np
from PIL import Image

def preprocess_img(img):
    gray = img.convert("L")
    img = np.array(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
    morphology_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    blur = cv2.GaussianBlur(morphology_img, (3, 3), 0)
    _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    padding = 5
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    cropped_image = binary[y:y + h, x:x + w]
    extra_space = np.full((cropped_image.shape[0] + 2 * padding, cropped_image.shape[1] + 2 * padding), 255, dtype=np.uint8)
    extra_space[padding:-padding, padding:-padding] = cropped_image
    corrected = cv2.resize(extra_space, (330, 175))
    resized_image = Image.fromarray(corrected)
    return resized_image
