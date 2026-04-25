import cv2
import numpy as np
import os

BASE_DIR = r"c:\Users\parra\Downloads\assignment\Alemeno Frontend Assignment Marker Images\Alemeno Frontend Assignment Marker Images"
OUTPUT_SIZE = 300

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def debug_image(image_path, label=""):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"\n--- {label} ---")
    print(f"  Total contours found: {len(contours)}")
    
    for i, cnt in enumerate(contours):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        if area > 500:
            ar = float(w)/h if h > 0 else 0
            print(f"  Contour {i}: vertices={len(approx)}, bbox=({x},{y},{w},{h}), area={area:.0f}, AR={ar:.2f}")

    # Also try RETR_TREE
    contours2, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  RETR_TREE contours: {len(contours2)}")
    for i, cnt in enumerate(contours2):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 500 and len(approx) == 4:
            ar = float(w)/h if h > 0 else 0
            parent = hier[0][i][3]
            print(f"  Tree-Contour {i}: vertices={len(approx)}, bbox=({x},{y},{w},{h}), area={area:.0f}, AR={ar:.2f}, parent={parent}")

# Marker 1 correct images that fail
for f in ["Marker1-TestImage2-Correct.jpg", "Marker1-TestImage3-Correct.jpg"]:
    path = os.path.join(BASE_DIR, "Marker1-TestImages", "Correct Marker Images", f)
    debug_image(path, f)

# Marker 2 correct images that fail
for f in ["Marker2-TestImage1-Correct.jpg", "Marker2-TestImage2-Correct.jpg", "Marker2-TestImage3-Correct.jpg"]:
    path = os.path.join(BASE_DIR, "Marker2-TestImages", "Correct Marker Images", f)
    debug_image(path, f)
