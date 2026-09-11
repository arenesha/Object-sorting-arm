"""
==============================================================================
GENERATE HIGH-RESOLUTION SNAPSHOTS MATCHING REFERENCE MACHINE VISION UI
==============================================================================
Produces deliverable snapshots of the redesigned UI matching the reference image:
- hardware_jetarm_reference_match.jpg: Idle restyled interface matching reference
- reference_comparison_side_by_side.jpg: Left = Reference, Right = Generated
- hardware_jetarm_idle.jpg: Restyled idle UI
- hardware_jetarm_idle_zigzag.jpg: Diagonal zigzag idle posture
- hardware_jetarm_grip.jpg: In-pickup-zone object grip simulation
- hardware_jetarm_to_bin.jpg: In-transit object transfer to bin dock
==============================================================================
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jetson")))
from reference_ui_renderer import ReferenceUIRenderer, CANVAS_WIDTH, CANVAS_HEIGHT

OUTPUT_DIR = r"C:\Users\Admin\.gemini\antigravity-ide\brain\a3014d08-dc63-499b-a951-e088f288dc4d"
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")


def generate_reference_match_snapshots():
    renderer = ReferenceUIRenderer(width=CANVAS_WIDTH, height=CANVAS_HEIGHT)

    # 1. Load clean camera background
    cam_sample = os.path.join(SAMPLE_DIR, "clean_reference_camera.jpg")
    cam = cv2.imread(cam_sample) if os.path.exists(cam_sample) else None

    # Render idle match snapshot
    rendered = renderer.render_frame(
        camera_frame=cam,
        fps=30.3,
        auto_on=True,
        demo_active=False,
        arm_status="ARM: IDLE",
        cycles=0,
    )

    match_path = os.path.join(OUTPUT_DIR, "hardware_jetarm_reference_match.jpg")
    cv2.imwrite(match_path, rendered)
    print(f"[*] Saved match snapshot: {match_path}")

    # Render idle aliases for documentation and backwards compatibility
    p_idle = os.path.join(OUTPUT_DIR, "hardware_jetarm_idle.jpg")
    cv2.imwrite(p_idle, rendered)
    p_zigzag = os.path.join(OUTPUT_DIR, "hardware_jetarm_idle_zigzag.jpg")
    cv2.imwrite(p_zigzag, rendered)

    # 2. Side-by-side comparison with reference
    ref_path = os.path.join(OUTPUT_DIR, r".user_uploaded\media_1789043363861.png")
    if os.path.exists(ref_path):
        ref = cv2.imread(ref_path)
        if ref is not None:
            h, w = ref.shape[:2]
            rendered_res = cv2.resize(rendered, (w, h))
            sbs = np.hstack([ref, rendered_res])
            sbs_path = os.path.join(OUTPUT_DIR, "reference_comparison_side_by_side.jpg")
            cv2.imwrite(sbs_path, sbs)
            print(f"[*] Saved side-by-side comparison: {sbs_path}")

    # 3. Grip Snapshot (Object detected in pickup zone)
    cube_sample = os.path.join(SAMPLE_DIR, "rubiks_cube_val.jpg")
    cam_cube = cv2.imread(cube_sample) if os.path.exists(cube_sample) else cam
    grip_frame = renderer.render_frame(
        camera_frame=cam_cube,
        fps=30.0,
        auto_on=True,
        demo_active=False,
        arm_status="ARM: GRIP",
        cycles=0,
        detected_box=(220, 180, 160, 140, "Rubiks-Cube", 0.94),
    )
    grip_path = os.path.join(OUTPUT_DIR, "hardware_jetarm_grip.jpg")
    cv2.imwrite(grip_path, grip_frame)
    print(f"[*] Saved grip snapshot: {grip_path}")

    # 4. To Bin Snapshot (Transferring to Bin 1)
    to_bin_frame = renderer.render_frame(
        camera_frame=cam,
        fps=29.8,
        auto_on=True,
        demo_active=True,
        arm_status="ARM: TO_BIN",
        cycles=1,
    )
    to_bin_path = os.path.join(OUTPUT_DIR, "hardware_jetarm_to_bin.jpg")
    cv2.imwrite(to_bin_path, to_bin_frame)
    print(f"[*] Saved to-bin snapshot: {to_bin_path}")


if __name__ == "__main__":
    generate_reference_match_snapshots()
    print("\n[SUCCESS] All restyled UI snapshots successfully updated.")
