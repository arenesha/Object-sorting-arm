"""Autonomous Object-Sorting Robotic Arm - Offline Vision Tuning & Test Tool

Allows tuning and evaluating HSV color thresholds and pickup zone ROI on saved images
or video recordings without requiring physical Jetson hardware or a live camera.

Modes:
  1. Automated Evaluation: Run against sample dataset and report detection accuracy.
  2. Interactive HSV Tuner: Real-time OpenCV GUI with sliders for H, S, V bounds and area.

Usage:
  python test_detection_offline.py                    # Run automated test on sample images
  python test_detection_offline.py --image <path>     # Run detection on a specific image
  python test_detection_offline.py --tune <image_path># Open interactive HSV trackbar tuner GUI
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Add jetson module directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jetson")))
from config import CONFIG, VisionConfig
from detector import ColorHSVDetector


def run_interactive_tuner(image_path: str):
    """Interactive GUI with sliders to dial in HSV color thresholding in real time."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return

    frame = cv2.imread(image_path)
    rx, ry, rw, rh = CONFIG.vision.roi_pickup_zone
    roi = frame[ry : ry + rh, rx : rx + rw]

    window_name = "HSV Threshold Tuner (Press 'q' or ESC to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 700)

    def nothing(x):
        pass

    # Trackbars for HSV range
    cv2.createTrackbar("H Min", window_name, 0, 179, nothing)
    cv2.createTrackbar("H Max", window_name, 179, 179, nothing)
    cv2.createTrackbar("S Min", window_name, 80, 255, nothing)
    cv2.createTrackbar("S Max", window_name, 255, 255, nothing)
    cv2.createTrackbar("V Min", window_name, 70, 255, nothing)
    cv2.createTrackbar("V Max", window_name, 255, 255, nothing)
    cv2.createTrackbar("Min Area (/100)", window_name, 12, 300, nothing)

    print("\n--- Interactive HSV Tuner Active ---")
    print("Adjust the sliders on the OpenCV window to isolate your target object.")
    print("Press 'q' or 'ESC' to close the tuner.\n")

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    while True:
        h_min = cv2.getTrackbarPos("H Min", window_name)
        h_max = cv2.getTrackbarPos("H Max", window_name)
        s_min = cv2.getTrackbarPos("S Min", window_name)
        s_max = cv2.getTrackbarPos("S Max", window_name)
        v_min = cv2.getTrackbarPos("V Min", window_name)
        v_max = cv2.getTrackbarPos("V Max", window_name)
        min_area = cv2.getTrackbarPos("Min Area (/100)", window_name) * 100

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        mask = cv2.inRange(hsv_roi, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Draw contour on ROI preview
        roi_preview = roi.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                cv2.drawContours(roi_preview, [cnt], -1, (0, 255, 0), 2)
                bx, by, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(roi_preview, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
                cv2.putText(
                    roi_preview,
                    f"Area: {int(area)}",
                    (bx, max(15, by - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

        # Tile display: Original ROI | Mask (colored) | Output Overlay
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        tiled = np.hstack([roi, mask_bgr, roi_preview])
        cv2.imshow(window_name, tiled)

        key = cv2.waitKey(30) & 0xFF
        if key in [ord("q"), 27]:
            break

    cv2.destroyAllWindows()
    print(f"\nFinal Slider Values to copy into config.py:")
    print(f"  Lower HSV: ({h_min}, {s_min}, {v_min})")
    print(f"  Upper HSV: ({h_max}, {s_max}, {v_max})")
    print(f"  Min Area:  {min_area}")


def evaluate_dataset(sample_dir: str, output_dir: str = None) -> bool:
    """Runs automated evaluation on all test images in the directory."""
    if not os.path.exists(sample_dir):
        print(f"Error: Sample directory not found at '{sample_dir}'")
        return False

    detector = ColorHSVDetector()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"  RUNNING OFFLINE DETECTION EVALUATION ON: {sample_dir}")
    print("=" * 70)

    # Expected outcomes mapping
    expectations = {
        "empty_tray.jpg": (None, None),
        "red_cube_center.jpg": ("red", 1),
        "green_cube_center.jpg": ("green", 2),
        "blue_cube_center.jpg": ("blue", 3),
        "red_cube_offset.jpg": ("red", 1),
        "green_cube_offset.jpg": ("green", 2),
        "blue_cube_offset.jpg": ("blue", 3),
        "red_cube_outside_roi.jpg": (None, None),  # Should NOT trigger
    }

    total = 0
    passed = 0

    image_files = sorted([f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png"))])

    for img_name in image_files:
        img_path = os.path.join(sample_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        total += 1
        result = detector.detect(frame)

        expected_class, expected_bin = expectations.get(img_name, (None, None))

        actual_class = result.class_name if result else None
        actual_bin = result.bin_id if result else None

        # Check if match
        is_correct = (actual_class == expected_class) and (actual_bin == expected_bin)
        if is_correct:
            passed += 1
            status_tag = "[PASS]"
        else:
            status_tag = "[FAIL]"

        print(
            f"{status_tag} {img_name:<26} | "
            f"Expected: {str(expected_class):<6} (Bin {str(expected_bin)}) | "
            f"Detected: {str(actual_class):<6} (Bin {str(actual_bin)})"
        )

        if output_dir:
            out_frame = result.annotated_frame if result else frame.copy()
            # Draw HUD tag if empty
            if not result:
                rx, ry, rw, rh = CONFIG.vision.roi_pickup_zone
                cv2.rectangle(out_frame, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 2)
                cv2.putText(
                    out_frame,
                    "NO OBJECT IN PICKUP ZONE",
                    (rx, ry - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 255),
                    1,
                )
            out_file = os.path.join(output_dir, f"det_{img_name}")
            cv2.imwrite(out_file, out_frame)

    print("-" * 70)
    print(f"Evaluation Complete: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    if output_dir:
        print(f"Annotated result images saved to: {output_dir}")
    print("=" * 70)
    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Offline Vision Detection & Tuning")
    parser.add_argument("--image", type=str, help="Path to single image file to test")
    parser.add_argument("--tune", type=str, help="Path to image for interactive HSV slider tuning")
    parser.add_argument(
        "--dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sample_images"),
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output_detected"),
        help="Directory to save annotated result images",
    )
    args = parser.parse_args()

    if args.tune:
        run_interactive_tuner(args.tune)
        return

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Failed to read image: {args.image}")
            return
        det = ColorHSVDetector()
        res = det.detect(frame)
        if res:
            print(f"Detected: {res.class_name.upper()} (Confidence: {res.confidence*100:.1f}%) -> BIN {res.bin_id}")
            print(f"Bounding Box: {res.bounding_box}, Center: {res.center}, Area: {res.area}")
        else:
            print("No object detected inside pickup zone.")
        return

    # Default: evaluate sample dataset
    success = evaluate_dataset(args.dir, args.output)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
