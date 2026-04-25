import cv2
import numpy as np
import os
import time

BASE_DIR = r"c:\Users\parra\Downloads\assignment\Alemeno Frontend Assignment Marker Images\Alemeno Frontend Assignment Marker Images"
OUTPUT_DIR = r"c:\Users\parra\Downloads\assignment\test_results"
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


def validate_marker1(warped):
    """
    Marker 1 structure (140x140):
    - Thick solid black border on all 4 sides
    - Small 20x20 black square in exactly ONE corner (top-left in canonical form)
    - The other 3 corners of the interior must be empty (white)
    - At least 60% interior area should be empty
    
    Key difference from incorrect: incorrect markers have the black square
    in the CENTER, not a corner.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    border = OUTPUT_SIZE // 7
    interior = thresh[border:OUTPUT_SIZE - border, border:OUTPUT_SIZE - border]
    ih, iw = interior.shape

    corner_size = ih // 4
    corners = [
        interior[0:corner_size, 0:corner_size],
        interior[0:corner_size, iw - corner_size:iw],
        interior[ih - corner_size:ih, iw - corner_size:iw],
        interior[ih - corner_size:ih, 0:corner_size],
    ]

    corner_fills = [np.mean(c) / 255.0 for c in corners]

    filled_corners = sum(1 for f in corner_fills if f > 0.3)
    if filled_corners != 1:
        return False, -1

    max_corner = np.argmax(corner_fills)
    if corner_fills[max_corner] < 0.3:
        return False, -1

    empty_corners = [i for i in range(4) if i != max_corner]
    for i in empty_corners:
        if corner_fills[i] > 0.15:
            return False, -1

    center_region = interior[ih // 4 : 3 * ih // 4, iw // 4 : 3 * iw // 4]
    center_fill = np.mean(center_region) / 255.0
    if center_fill > 0.25:
        return False, -1

    total_interior_fill = np.mean(interior) / 255.0
    if total_interior_fill > 0.35:
        return False, -1

    return True, max_corner


def validate_marker2(warped):
    """
    Marker 2 structure (160x160):
    - L-shaped solid border: left side + bottom side are thick solid black
    - Top and right sides are dashed (series of small black squares)
    - Interior is mostly empty
    
    Key difference from incorrect: incorrect markers have wrong combinations 
    of solid/dashed sides or all solid sides.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    border_width = OUTPUT_SIZE // 10
    edge_depth = border_width

    top_edge = thresh[0:edge_depth, border_width:OUTPUT_SIZE - border_width]
    bottom_edge = thresh[OUTPUT_SIZE - edge_depth:OUTPUT_SIZE, border_width:OUTPUT_SIZE - border_width]
    left_edge = thresh[border_width:OUTPUT_SIZE - border_width, 0:edge_depth]
    right_edge = thresh[border_width:OUTPUT_SIZE - border_width, OUTPUT_SIZE - edge_depth:OUTPUT_SIZE]

    top_fill = np.mean(top_edge) / 255.0
    bottom_fill = np.mean(bottom_edge) / 255.0
    left_fill = np.mean(left_edge) / 255.0
    right_fill = np.mean(right_edge) / 255.0

    edges = [top_fill, right_fill, bottom_fill, left_fill]

    solid_threshold = 0.7
    dashed_low = 0.2
    dashed_high = 0.65

    solid_count = sum(1 for e in edges if e > solid_threshold)
    dashed_count = sum(1 for e in edges if dashed_low < e < dashed_high)

    if solid_count != 2 or dashed_count != 2:
        return False, -1

    solid_edges = [i for i, e in enumerate(edges) if e > solid_threshold]
    if not (
        (solid_edges == [2, 3]) or
        (solid_edges == [0, 3]) or
        (solid_edges == [0, 1]) or
        (solid_edges == [1, 2])
    ):
        return False, -1

    if solid_edges == [2, 3]:
        orientation = 0
    elif solid_edges == [0, 3]:
        orientation = 1
    elif solid_edges == [0, 1]:
        orientation = 2
    elif solid_edges == [1, 2]:
        orientation = 3
    else:
        return False, -1

    return True, orientation


def correct_orientation_m1(warped, corner_idx):
    if corner_idx == 0:
        return warped
    elif corner_idx == 1:
        return cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif corner_idx == 2:
        return cv2.rotate(warped, cv2.ROTATE_180)
    elif corner_idx == 3:
        return cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def correct_orientation_m2(warped, orientation):
    if orientation == 0:
        return warped
    elif orientation == 1:
        return cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 2:
        return cv2.rotate(warped, cv2.ROTATE_180)
    elif orientation == 3:
        return cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return warped


def extract_square(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2 and w > 50:
                pts = np.float32([approx[i][0] for i in range(4)])
                rect = order_points(pts)
                dst = np.float32(
                    [[0, 0], [OUTPUT_SIZE, 0], [OUTPUT_SIZE, OUTPUT_SIZE], [0, OUTPUT_SIZE]]
                )
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img, M, (OUTPUT_SIZE, OUTPUT_SIZE))
                candidates.append((warped, cv2.contourArea(cnt)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def detect_marker(image_path, marker_type=1):
    warped = extract_square(image_path)
    if warped is None:
        return None

    if marker_type == 1:
        valid, orient = validate_marker1(warped)
        if not valid:
            return None
        corrected = correct_orientation_m1(warped, orient)
        return corrected, orient
    else:
        valid, orient = validate_marker2(warped)
        if not valid:
            return None
        corrected = correct_orientation_m2(warped, orient)
        return corrected, orient


def run_tests():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("ALEMENO MARKER DETECTION - TEST SUITE")
    print("=" * 60)

    total_passed = 0
    total_failed = 0

    for marker_num in [1, 2]:
        marker_dir = os.path.join(BASE_DIR, f"Marker{marker_num}-TestImages")
        if not os.path.isdir(marker_dir):
            print(f"\n[SKIP] Marker{marker_num} test images not found")
            continue

        print(f"\n{'-' * 40}")
        print(f"  MARKER {marker_num}")
        print(f"{'-' * 40}")

        correct_dir = os.path.join(marker_dir, "Correct Marker Images")
        incorrect_dir = os.path.join(marker_dir, "Incorrect Marker Images")

        passed = 0
        failed = 0

        if os.path.isdir(correct_dir):
            print("\n  [+] CORRECT images (should be DETECTED):")
            for f in sorted(os.listdir(correct_dir)):
                if not f.lower().endswith((".jpg", ".png")):
                    continue
                path = os.path.join(correct_dir, f)
                start = time.time()
                result = detect_marker(path, marker_num)
                elapsed = (time.time() - start) * 1000

                if result is not None:
                    img, orient = result
                    out_path = os.path.join(OUTPUT_DIR, f"M{marker_num}_{f}")
                    cv2.imwrite(out_path, img)
                    status = "PASS"
                    detail = f"orient={orient}, {img.shape[1]}x{img.shape[0]}px, {elapsed:.0f}ms"
                    passed += 1
                else:
                    status = "FAIL"
                    detail = f"NOT detected ({elapsed:.0f}ms)"
                    failed += 1
                print(f"    [{status}] {f}")
                print(f"             {detail}")

        if os.path.isdir(incorrect_dir):
            print(f"\n  [-] INCORRECT images (should be REJECTED):")
            for f in sorted(os.listdir(incorrect_dir)):
                if not f.lower().endswith((".jpg", ".png")):
                    continue
                path = os.path.join(incorrect_dir, f)
                start = time.time()
                result = detect_marker(path, marker_num)
                elapsed = (time.time() - start) * 1000

                if result is None:
                    status = "PASS"
                    detail = f"correctly rejected ({elapsed:.0f}ms)"
                    passed += 1
                else:
                    status = "FAIL"
                    detail = f"FALSE POSITIVE ({elapsed:.0f}ms)"
                    failed += 1
                print(f"    [{status}] {f}")
                print(f"             {detail}")

        total_passed += passed
        total_failed += failed
        print(f"\n  Marker {marker_num}: {passed}/{passed + failed} tests passed")

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {total_passed}/{total_passed + total_failed} tests passed")
    if total_failed == 0:
        print("  ALL TESTS PASSED!")
    else:
        print(f"  {total_failed} test(s) failed")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_tests()
