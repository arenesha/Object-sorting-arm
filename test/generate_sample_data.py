"""Autonomous Object-Sorting Robotic Arm - Synthetic Test Data Generator

Generates realistic camera test images simulating the pickup zone and sample objects
(red cube, green cube, blue cube, empty tray, and distractors).
These allow tuning and verifying HSV vision detection completely offline without
needing the physical Jetson or camera connected.
"""

import os
import sys
import cv2
import numpy as np

# Add jetson module directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jetson")))
from config import CONFIG


def create_tray_background(width=640, height=480) -> np.ndarray:
    """Creates a realistic work surface background with ambient lighting and grid."""
    # Base workbench neutral grey-beige
    bg = np.full((height, width, 3), (195, 205, 210), dtype=np.uint8)

    # Add subtle texture noise
    noise = np.random.normal(0, 3, (height, width, 3)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw tabletop grid lines
    grid_color = (180, 190, 195)
    for x in range(0, width, 40):
        cv2.line(bg, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, 40):
        cv2.line(bg, (0, y), (width, y), grid_color, 1)

    # Draw the physical pickup tray outline
    rx, ry, rw, rh = CONFIG.vision.roi_pickup_zone
    tray_margin = 15
    tx, ty, tw, th = rx - tray_margin, ry - tray_margin, rw + (tray_margin * 2), rh + (tray_margin * 2)

    # White tray surface
    cv2.rectangle(bg, (tx, ty), (tx + tw, ty + th), (240, 240, 245), -1)
    # Tray beveled border
    cv2.rectangle(bg, (tx, ty), (tx + tw, ty + th), (150, 150, 160), 3)

    # Tape / marked alignment corners for pickup zone
    corner_len = 15
    marker_color = (70, 70, 70)
    # Top-Left
    cv2.line(bg, (rx, ry), (rx + corner_len, ry), marker_color, 2)
    cv2.line(bg, (rx, ry), (rx, ry + corner_len), marker_color, 2)
    # Top-Right
    cv2.line(bg, (rx + rw, ry), (rx + rw - corner_len, ry), marker_color, 2)
    cv2.line(bg, (rx + rw, ry), (rx + rw, ry + corner_len), marker_color, 2)
    # Bottom-Left
    cv2.line(bg, (rx, ry + rh), (rx + corner_len, ry + rh), marker_color, 2)
    cv2.line(bg, (rx, ry + rh), (rx, ry + rh - corner_len), marker_color, 2)
    # Bottom-Right
    cv2.line(bg, (rx + rw, ry + rh), (rx + rw - corner_len, ry + rh), marker_color, 2)
    cv2.line(bg, (rx + rw, ry + rh), (rx + rw, ry + rh - corner_len), marker_color, 2)

    # Text label on physical tray
    cv2.putText(
        bg,
        "SORTING TRAY - PICKUP ZONE",
        (tx + 10, ty + th + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 100, 110),
        1,
        cv2.LINE_AA,
    )
    return bg


def render_3d_cube(
    img: np.ndarray,
    center_x: int,
    center_y: int,
    size: int,
    base_bgr: tuple,
) -> np.ndarray:
    """Renders a simulated colored sorting block with top/side 3D shading."""
    out = img.copy()
    half = size // 2
    offset = int(size * 0.3)

    # Shading factors for isometric look
    top_color = tuple(min(255, int(c * 1.25)) for c in base_bgr)
    side_color = tuple(max(0, int(c * 0.75)) for c in base_bgr)
    front_color = base_bgr

    # Front face
    fx1, fy1 = center_x - half, center_y - half + offset
    fx2, fy2 = center_x + half, center_y + half + offset
    cv2.rectangle(out, (fx1, fy1), (fx2, fy2), front_color, -1)
    cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (40, 40, 40), 1)

    # Top face (polygon)
    top_pts = np.array([
        [fx1, fy1],
        [fx1 + offset, fy1 - offset],
        [fx2 + offset, fy1 - offset],
        [fx2, fy1],
    ], dtype=np.int32)
    cv2.fillPoly(out, [top_pts], top_color)
    cv2.polylines(out, [top_pts], True, (40, 40, 40), 1)

    # Side face (polygon)
    side_pts = np.array([
        [fx2, fy1],
        [fx2 + offset, fy1 - offset],
        [fx2 + offset, fy2 - offset],
        [fx2, fy2],
    ], dtype=np.int32)
    cv2.fillPoly(out, [side_pts], side_color)
    cv2.polylines(out, [side_pts], True, (40, 40, 40), 1)

    return out


def generate_dataset(output_dir: str):
    """Generates a complete set of synthetic evaluation images."""
    os.makedirs(output_dir, exist_ok=True)
    rx, ry, rw, rh = CONFIG.vision.roi_pickup_zone
    center_x = rx + (rw // 2)
    center_y = ry + (rh // 2)

    samples = [
        # 1. Empty tray
        ("empty_tray.jpg", None, None),
        # 2. Red object centered
        ("red_cube_center.jpg", (20, 20, 215), (center_x, center_y)),
        # 3. Green object centered
        ("green_cube_center.jpg", (30, 185, 30), (center_x, center_y)),
        # 4. Blue object centered
        ("blue_cube_center.jpg", (210, 80, 20), (center_x, center_y)),
        # 5. Red object slightly offset in pickup zone
        ("red_cube_offset.jpg", (15, 25, 230), (center_x - 30, center_y + 20)),
        # 6. Green object slightly offset
        ("green_cube_offset.jpg", (35, 175, 45), (center_x + 25, center_y - 20)),
        # 7. Blue object slightly offset
        ("blue_cube_offset.jpg", (220, 95, 35), (center_x - 20, center_y - 30)),
        # 8. Object OUTSIDE pickup zone (distractor - should be ignored)
        ("red_cube_outside_roi.jpg", (20, 20, 215), (rx - 100, ry)),
    ]

    print(f"Generating synthetic test dataset in '{output_dir}'...")
    for filename, color_bgr, pos in samples:
        frame = create_tray_background()
        if color_bgr and pos:
            frame = render_3d_cube(frame, pos[0], pos[1], size=65, base_bgr=color_bgr)

        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"  -> Generated: {filename}")

    print(f"Done! {len(samples)} sample images created successfully.")


if __name__ == "__main__":
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "sample_images"))
    generate_dataset(out_path)
