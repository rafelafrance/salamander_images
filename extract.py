#!/usr/bin/env python3
"""Extract individual salamanders from pictures of many of them."""

from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2


IN_DIR = Path('.') / 'data' / 'P_cinereus'
OUT_DIR = Path('.') / 'data' / 'P_cinereus_segmented'

SIZE = 2048 + 512
COLOR = (0, 0, 255)


def crop(hull, img):
    """Crop the image to include the hull."""
    x, y, width, height = cv2.boundingRect(hull)

    rem_h = SIZE - height
    rem_w = SIZE - width

    if width > SIZE:
        x = x - (rem_w // 2)
        width = SIZE

    if height > SIZE:
        y = y - (rem_h // 2)
        height = SIZE

    img = img[y:y+height, x:x+width]

    rem_h = SIZE - height
    rem_w = SIZE - width

    top = rem_h // 2
    bottom = rem_h - top
    left = rem_w // 2
    right = rem_w - left

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return img


def get_edges(original):
    """Get the image edges for extracting labels."""
    height = original.shape[0] // 10
    width = original.shape[1] // 10

    kernel = np.ones((8, 8), np.uint8)

    gray = cv2.resize(original, (width, height))
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 320, 100)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    edges = cv2.resize(edges, (original.shape[1], original.shape[0]))

    return edges


def color_filter(original):
    """Filter the image by color to extract the salamanders."""
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    low = np.array([0, 90, 0])
    high = np.array([100, 255, 255])

    filtered = cv2.inRange(hsv, low, high)

    kernel = np.ones((128, 128), np.uint8)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    filtered = cv2.dilate(filtered, kernel, iterations=1)

    return filtered


def find_hulls(mask):
    """Build convex hulls around the salamanders and labels."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hulls = [cv2.convexHull(c, False) for c in contours]
    hulls = sorted(hulls, key=lambda h: cv2.boundingRect(h)[:2][::-1])
    return hulls


def has_salamander(image, filtered):
    """Check that the hull has a salamander."""
    overlap = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    overlap = cv2.bitwise_and(overlap, overlap, mask=filtered)
    _, overlap = cv2.threshold(overlap, 16, 255, cv2.THRESH_BINARY)
    overlap = overlap * (1 / 255)
    return overlap.sum() > 100000.0


def output_salamanders(original, filtered, hulls, path):
    """Output the extracted salamanders."""
    # original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    idx = 0
    for i, hull in enumerate(hulls):
        image = original.copy()

        area = cv2.contourArea(hull)
        if area < 100_000:
            continue

        cv2.drawContours(image, hulls, i, COLOR, thickness=-1, lineType=8)
        mask = cv2.inRange(image, COLOR, COLOR)

        image = original.copy()
        image = cv2.bitwise_and(image, image, mask=mask)

        if not has_salamander(image, filtered):
            continue

        image = crop(hull, image)

        idx += 1
        item = str(idx).zfill(2)
        file_name = str(path) + f'_{item}.jpg'

        cv2.imwrite(file_name, image)


def extract():
    """Extract the salamanders."""
    images = sorted(IN_DIR.glob('**/*.[Jj][Pp][Gg]'))
    for image in tqdm(images):
        path = Path(image)
        path = OUT_DIR / '_'.join([path.parent.stem, path.stem])
        original = cv2.imread(str(image))
        edges = get_edges(original)
        filtered = color_filter(original)
        mask = cv2.bitwise_or(edges, filtered)
        hulls = find_hulls(mask)
        output_salamanders(original, filtered, hulls, path)


if __name__ == '__main__':
    extract()
