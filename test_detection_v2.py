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


def extract_square_candidates(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    candidates = []

    for block_size in [11, 21, 31]:
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 2
        )

        thresh_variants = [thresh]
        for ksize in [15, 25, 35]:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh_variants.append(closed)

        for t in thresh_variants:
            for mode in [cv2.RETR_EXTERNAL, cv2.RETR_TREE]:
                contours, _ = cv2.findContours(t, mode, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect_ratio = float(w) / h
                        area = cv2.contourArea(cnt)
                        if 0.85 <= aspect_ratio <= 1.15 and area > 2000:
                            pts = np.float32([approx[i][0] for i in range(4)])
                            rect = order_points(pts)
                            dst = np.float32(
                                [[0, 0], [OUTPUT_SIZE, 0], [OUTPUT_SIZE, OUTPUT_SIZE], [0, OUTPUT_SIZE]]
                            )
                            M = cv2.getPerspectiveTransform(rect, dst)
                            warped = cv2.warpPerspective(img, M, (OUTPUT_SIZE, OUTPUT_SIZE))
                            candidates.append((warped, area))

    seen = set()
    unique = []
    for warped, area in candidates:
        key = int(area)
        if key not in seen:
            seen.add(key)
            unique.append((warped, area))

    unique.sort(key=lambda x: x[1], reverse=True)
    return unique


def validate_marker1(warped):
    """
    Marker 1: 140x140 square with solid black border on all 4 sides.
    Small 20x20 orientation square in exactly ONE corner.
    
    The orientation square should be ~14% of the marker side (20/140).
    Incorrect markers have the square in the center or much too large.
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

    filled_corners = sum(1 for f in corner_fills if f > 0.25)
    if filled_corners != 1:
        return False, -1

    max_corner = np.argmax(corner_fills)
    if corner_fills[max_corner] < 0.25:
        return False, -1

    empty_corners = [i for i in range(4) if i != max_corner]
    for i in empty_corners:
        if corner_fills[i] > 0.12:
            return False, -1

    if corner_fills[max_corner] > 0.85:
        return False, -1

    small_corner = corner_size // 2
    corner_coords = [
        (0, 0),
        (0, iw - small_corner),
        (ih - small_corner, iw - small_corner),
        (ih - small_corner, 0),
    ]
    r, c = corner_coords[max_corner]
    tight_corner = interior[r:r + small_corner, c:c + small_corner]
    tight_fill = np.mean(tight_corner) / 255.0
    if tight_fill < 0.4:
        return False, -1

    has_solid_border = True
    edge_strip = OUTPUT_SIZE // 15
    top_border = thresh[0:edge_strip, :]
    bottom_border = thresh[OUTPUT_SIZE - edge_strip:OUTPUT_SIZE, :]
    left_border = thresh[:, 0:edge_strip]
    right_border = thresh[:, OUTPUT_SIZE - edge_strip:OUTPUT_SIZE]

    border_fills = [
        np.mean(top_border) / 255.0,
        np.mean(right_border) / 255.0,
        np.mean(bottom_border) / 255.0,
        np.mean(left_border) / 255.0,
    ]
    for bf in border_fills:
        if bf < 0.5:
            has_solid_border = False
            break

    if not has_solid_border:
        return False, -1

    return True, max_corner


def validate_marker2(warped):
    """
    Marker 2: 160x160 square.
    L-shaped border: 2 adjacent sides are solid black, 2 adjacent sides are dashed.
    Canonical form: left + bottom solid, top + right dashed.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    strip = OUTPUT_SIZE // 12

    top = thresh[0:strip, strip:OUTPUT_SIZE - strip]
    bottom = thresh[OUTPUT_SIZE - strip:OUTPUT_SIZE, strip:OUTPUT_SIZE - strip]
    left = thresh[strip:OUTPUT_SIZE - strip, 0:strip]
    right = thresh[strip:OUTPUT_SIZE - strip, OUTPUT_SIZE - strip:OUTPUT_SIZE]

    fills = [
        np.mean(top) / 255.0,
        np.mean(right) / 255.0,
        np.mean(bottom) / 255.0,
        np.mean(left) / 255.0,
    ]

    sorted_indices = sorted(range(4), key=lambda i: fills[i], reverse=True)
    top2 = set(sorted_indices[:2])
    bot2 = set(sorted_indices[2:])

    for i in top2:
        if fills[i] < 0.55:
            return False, -1
    for i in bot2:
        if fills[i] > 0.70 or fills[i] < 0.10:
            return False, -1

    gap = fills[sorted_indices[1]] - fills[sorted_indices[2]]
    if gap < 0.05:
        return False, -1

    solid_indices = top2
    dashed_indices = bot2

    solid_count = 2
    dashed_count = 2

    if solid_count != 2 or dashed_count != 2:
        return False, -1


    adjacent_pairs = [{0, 3}, {0, 1}, {1, 2}, {2, 3}]
    if solid_indices not in adjacent_pairs:
        return False, -1
    if dashed_indices not in adjacent_pairs:
        return False, -1

    if solid_indices == {2, 3}:
        orientation = 0
    elif solid_indices == {0, 3}:
        orientation = 1
    elif solid_indices == {0, 1}:
        orientation = 2
    elif solid_indices == {1, 2}:
        orientation = 3
    else:
        return False, -1

    return True, orientation


def correct_orientation_m1(warped, corner_idx):
    rotations = {
        0: None,
        1: cv2.ROTATE_90_COUNTERCLOCKWISE,
        2: cv2.ROTATE_180,
        3: cv2.ROTATE_90_CLOCKWISE,
    }
    r = rotations.get(corner_idx)
    return cv2.rotate(warped, r) if r is not None else warped


def correct_orientation_m2(warped, orientation):
    rotations = {
        0: None,
        1: cv2.ROTATE_90_CLOCKWISE,
        2: cv2.ROTATE_180,
        3: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    r = rotations.get(orientation)
    return cv2.rotate(warped, r) if r is not None else warped


def detect_marker(image_path, marker_type=1):
    candidates = extract_square_candidates(image_path)
    if not candidates:
        return None

    for warped, area in candidates:
        if marker_type == 1:
            valid, orient = validate_marker1(warped)
            if valid:
                return correct_orientation_m1(warped, orient), orient
        else:
            valid, orient = validate_marker2(warped)
            if valid:
                return correct_orientation_m2(warped, orient), orient

    return None


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
