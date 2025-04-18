import cv2
import numpy as np
from PIL import Image

def preprocess_img(img_path):
    # Step 1: Convert Image to Grayscale
    gray = img_path.convert("L")  # Convert to grayscale

    # Step 2: Convert Grayscale to NumPy Array
    img = np.array(gray)

    # Step 3: Morphological Transformation (Enhancing text regions)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
    morphology_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 4: Apply Gaussian Blur (Smoothen the image)
    blur = cv2.GaussianBlur(morphology_img, (3, 3), 0)

    # Step 5: Detect Background Color & Apply Correct Thresholding
    mean_intensity = np.mean(blur)
    if mean_intensity < 127:
        # Dark background → Use THRESH_BINARY_INV to make text black
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        # Light background → Use THRESH_BINARY to keep text black
        _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Find the Bounding Box Around Non-White Pixels (Text Area)
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)

    # Step 7: Adding Padding to the Bounding Box
    padding = 5
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding

    # Ensure the coordinates remain within the image boundaries
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)

    # Step 8: Crop and Add Extra White Space
    cropped_image = binary[y:y + h, x:x + w]
    extra_space = np.ones((cropped_image.shape[0] + 2 * padding, cropped_image.shape[1] + 2 * padding), dtype=np.uint8) * 255
    extra_space[padding:-padding, padding:-padding] = cropped_image

    # Step 9: Resize Image to Standard Size
    corrected = cv2.resize(extra_space, (330, 175))

    # Step 10: Convert the NumPy Array Back to a PIL Image
    resized_image = Image.fromarray(corrected)

    return resized_image