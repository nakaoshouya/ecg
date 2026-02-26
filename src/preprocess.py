import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image

def delete_red(img):
    b, g, r = cv2.split(img)
    mask = (r.astype(int) - g.astype(int) > 40) & (r.astype(int) - b.astype(int) > 25)
    img[mask] = [254, 254, 254]
    return img

def change_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def find_QR(img):
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    height, width = img.shape[:2]

    if points is not None and len(points) > 0:
        pts = points[0]
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())

        if cx < width / 2 and cy < height / 2:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif cx >= width / 2 and cy < height / 2:
            return img
        elif cx < width / 2 and cy >= height / 2:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
            return rotated
        else:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        if (height > width) and (height != 4000) and (width != 3000):
            plt.imshow(img, cmap="gray")
            plt.axis("off")
            plt.show()
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return img

def sizing(img):
    img = cv2.resize(img, (1024, 768))
    return img

def remove_black_vertical_lines(img, black_thresh=120, max_aspect=10.0, min_height_ratio=0.6, max_width=4):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    H, W = gray.shape
    bw = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)[1]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = gray.copy()
    for lab in range(1, num_labels):
        x, y, w, h, area = stats[lab]
        aspect = h / max(w, 1)
        if aspect > max_aspect and h >= int(H * min_height_ratio) and w <= max_width:
            out[labels == lab] = 255
    return out

def preprocess_test_image(img_path):
    img = cv2.imread(str(img_path))
    height, width = img.shape[:2]
    img = find_QR(img)

    if (height == 4000) and (width == 3000):
        img = img[1400:height-250, :]
        img = change_gray(img)
        img = sizing(img)
        img = remove_black_vertical_lines(img, black_thresh=150, max_aspect=3.0, min_height_ratio=0.6, max_width=10)
    else:
        img = delete_red(img)
        img = change_gray(img)
        img = sizing(img)

    return img
