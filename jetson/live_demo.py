"""
==============================================================================
AUTONOMOUS OBJECT-SORTING ROBOTIC ARM - LIVE DETECTION DEMO MODE
==============================================================================
NOTE FOR STUDENTS, RESEARCHERS & OPERATORS:
------------------------------------------------------------------------------
This script (live_demo.py) is a VISUALIZATION & SIMULATION tool ONLY.
It provides an interactive, mouse-driven demonstration of the live vision
pipeline (HSV or YOLOv8) and simulated sorting without needing physical arm
hardware attached.

WHEN PHYSICAL ROBOT HARDWARE IS CONNECTED:
------------------------------------------------------------------------------
Do NOT use this demo script for hardware operations.
Instead, run:
    python jetson/main.py       (CLI autonomous orchestrator)
or:
    python dashboard.py         (Full Web GUI & Telemetry Dashboard)

`main.py` is the official master orchestrator that interfaces with
`arm_controller.py`, drives the PCA9685 I2C servo controller, and carries
out physical autonomous pick-and-place cycles with real objects.
==============================================================================
"""

import os
import sys
import time
import threading
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure jetson module directory is in Python path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JETSON_DIR = os.path.join(BASE_DIR, "jetson")
if JETSON_DIR not in sys.path:
    sys.path.insert(0, JETSON_DIR)

from config import CONFIG, VisionConfig, get_active_model_info
from detector import BaseDetector, ColorHSVDetector, YOLODetector, DetectionResult
from logger import SortLogger
from jetarm_overlay import (
    JetArmOverlay,
    draw_rounded_rect,
    draw_corner_brackets,
    draw_hud_badge,
    UI_BG_DARK,
    UI_PANEL_BG,
    UI_PANEL_HEADER,
    UI_PANEL_BORDER,
    UI_PANEL_BORDER_LIGHT,
    UI_RECEPTACLE_BG,
    UI_TEXT_TITLE,
    UI_TEXT_BODY,
    UI_TEXT_MUTED,
    UI_ACCENT_CYAN,
    UI_ACCENT_GREEN,
    UI_ACCENT_AMBER,
    UI_ACCENT_RED,
    UI_ACCENT_BLUE,
)


# ==============================================================================
# ASYNCHRONOUS VISION WORKER FOR REAL-TIME 25-30+ FPS DISPLAY
# ==============================================================================
class AsyncVisionWorker:
    """
    Runs YOLOv8/HSV detector inference asynchronously in a dedicated daemon
    thread. This prevents CPU neural network latency from stalling camera
    capture and OpenCV display, keeping the live feed at solid 25-30+ FPS.
    """

    def __init__(self, detector: BaseDetector):
        self.detector = detector
        self.latest_result: Optional[DetectionResult] = None
        self.latest_time: float = 0.0
        self.frame_to_process: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _worker_loop(self):
        while self.running:
            got_event = self.new_frame_event.wait(timeout=0.1)
            if not self.running:
                break
            if not got_event:
                continue

            with self.lock:
                frame = self.frame_to_process
                self.frame_to_process = None
                self.new_frame_event.clear()

            if frame is not None:
                try:
                    result = self.detector.detect(frame)
                    with self.lock:
                        self.latest_result = result
                        self.latest_time = time.time()
                except Exception:
                    pass

    def submit_frame(self, frame: np.ndarray):
        """Submit frame only when worker is ready. Drops frames when busy to prevent lag queues."""
        if not self.new_frame_event.is_set():
            with self.lock:
                self.frame_to_process = frame.copy()
                self.new_frame_event.set()

    def get_result(self, max_age_sec: float = 1.0) -> Optional[DetectionResult]:
        """Returns the latest detection if it was produced within max_age_sec."""
        with self.lock:
            if self.latest_result is not None and (time.time() - self.latest_time <= max_age_sec):
                return self.latest_result
            return None

    def stop(self):
        self.running = False
        self.new_frame_event.set()


# ==============================================================================
# CONFIGURABLE LAYOUT CONSTANTS
# ==============================================================================
# Frame dimensions (matches camera pipeline)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Pickup Zone ROI (read directly from existing config.py)
# (rx, ry, rw, rh) -> Centered pickup workspace
PICKUP_ROI = CONFIG.vision.roi_pickup_zone

# Configurable Bin Zone Cards in the upper region above the pickup zone
BIN_ZONES: Dict[int, Dict[str, any]] = {
    1: {
        "name": "BIN 1",
        "tag": "LEFT",
        "rect": (20, 46, 186, 144),
        "base_color": UI_ACCENT_RED,         # Neon Crimson (75, 75, 245)
        "highlight_color": UI_ACCENT_GREEN,  # Cyber Emerald (100, 225, 45)
    },
    2: {
        "name": "BIN 2",
        "tag": "CENTER",
        "rect": (227, 46, 186, 144),
        "base_color": UI_ACCENT_GREEN,       # Cyber Emerald (100, 225, 45)
        "highlight_color": UI_ACCENT_GREEN,
    },
    3: {
        "name": "BIN 3",
        "tag": "RIGHT",
        "rect": (434, 46, 186, 144),
        "base_color": UI_ACCENT_BLUE,        # Electric Cobalt (250, 150, 30)
        "highlight_color": UI_ACCENT_GREEN,
    },
}


# ==============================================================================
# INTERACTIVE MOUSE DRAG STATE CONTAINER
# ==============================================================================
class DragSimulationState:
    """Encapsulates mouse drag-and-drop state for manual sort simulation."""

    def __init__(self):
        self.is_dragging = False
        self.mouse_pos = (0, 0)
        self.drag_offset = (0, 0)          # Offset from mouse to top-left of box
        self.drag_rect = (0, 0, 0, 0)      # (x, y, w, h) current position
        self.original_rect = (0, 0, 0, 0)  # Starting box in pickup zone
        self.drag_class = ""
        self.drag_conf = 0.0
        self.drag_bin_target = 0
        self.drag_patch: Optional[np.ndarray] = None
        self.drag_start_time = 0.0

        # Visual feedback timers
        self.flash_bin_id: Optional[int] = None
        self.flash_until: float = 0.0
        self.status_message: str = ""
        self.status_until: float = 0.0
        self.status_color: Tuple[int, int, int] = (0, 255, 0)

    def trigger_success(self, bin_id: int, class_name: str):
        """Trigger visual confirmation upon valid bin drop."""
        now = time.time()
        self.flash_bin_id = bin_id
        self.flash_until = now + 1.8
        self.status_message = f"SORTED: '{class_name.upper()}' -> BIN {bin_id} [SIMULATED]"
        self.status_color = UI_ACCENT_GREEN
        self.status_until = now + 2.5

    def trigger_snap_back(self):
        """Trigger feedback when dropped outside bins."""
        now = time.time()
        self.status_message = "DROPPED OUTSIDE BINS -> Snapped back to Pickup Zone"
        self.status_color = UI_ACCENT_AMBER
        self.status_until = now + 1.5


DRAG_STATE = DragSimulationState()
ACTIVE_DETECTION: Optional[DetectionResult] = None
JETARM: Optional[JetArmOverlay] = None
AUTOSORT_BTN_RECT = (268, 7, 126, 24)    # (x, y, w, h) Clickable HUD Auto-Sort Toggle button
DEMO_BTN_RECT = (404, 7, 114, 24)        # (x, y, w, h) Clickable HUD Demo Sort button


def draw_dashed_rect(
    canvas: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    dash_len: int = 5,
    gap_len: int = 4,
    thickness: int = 1,
):
    """Draws a clean dashed bounding rectangle using OpenCV lines."""
    x1, y1 = pt1
    x2, y2 = pt2
    for x in range(x1, x2, dash_len + gap_len):
        cv2.line(canvas, (x, y1), (min(x + dash_len, x2), y1), color, thickness, cv2.LINE_AA)
    for x in range(x1, x2, dash_len + gap_len):
        cv2.line(canvas, (x, y2), (min(x + dash_len, x2), y2), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash_len + gap_len):
        cv2.line(canvas, (x1, y), (x1, min(y + dash_len, y2)), color, thickness, cv2.LINE_AA)
    for y in range(y1, y2, dash_len + gap_len):
        cv2.line(canvas, (x2, y), (x2, min(y + dash_len, y2)), color, thickness, cv2.LINE_AA)


# ==============================================================================
# POLISHED INDUSTRIAL DOCKING BAY BIN ZONE RENDERER
# ==============================================================================
def draw_bin_card(
    canvas: np.ndarray,
    bin_id: int,
    bin_info: Dict[str, any],
    is_flashing: bool,
    is_incoming: bool,
):
    """
    Renders an industrial cyber-physical bin docking bay matching the reference interface:
    - Glowing category color border with smooth rounded corners
    - Header with solid category dot, bold white title, and right location pill badge
    - Inner recessed drop dock with dashed border, downward double chevrons, and labels
    """
    bx, by, bw, bh = bin_info["rect"]
    base_col = bin_info["base_color"]

    # 1. Dynamic border and fill based on interaction state
    if is_flashing:
        card_fill = (18, 42, 26)
        border_col = UI_ACCENT_GREEN
    elif is_incoming:
        card_fill = (26, 38, 50)
        border_col = UI_ACCENT_CYAN
    else:
        card_fill = (22, 16, 12)
        border_col = base_col

    # 2. Outer Card Body (Smooth rounded rectangle with glowing border)
    draw_rounded_rect(
        canvas,
        (bx, by),
        (bx + bw, by + bh),
        color=border_col,
        thickness=1,
        radius=12,
        fill_color=card_fill,
    )

    # 3. Header Indicators & Typography
    # Glowing category color dot
    cv2.circle(canvas, (bx + 18, by + 20), 5, border_col, -1, cv2.LINE_AA)

    # Title: Bold white
    cv2.putText(
        canvas,
        bin_info["name"],
        (bx + 32, by + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Location Tag in micro pill badge
    tag_str = bin_info.get("tag", "")
    if tag_str:
        (tw, _), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        tag_w = tw + 14
        tag_x = bx + bw - tag_w - 12
        pill_col = UI_ACCENT_GREEN if is_flashing else (UI_ACCENT_CYAN if is_incoming else base_col)
        draw_rounded_rect(
            canvas,
            (tag_x, by + 12),
            (tag_x + tag_w, by + 28),
            color=pill_col,
            thickness=1,
            radius=4,
            fill_color=(28, 18, 14),
        )
        cv2.putText(
            canvas,
            tag_str,
            (tag_x + 7, by + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            pill_col,
            1,
            cv2.LINE_AA,
        )

    # 4. Inner Recessed Drop Zone Docking Bay
    rx_d = bx + 10
    ry_d = by + 42
    rw_d = bw - 20
    rh_d = bh - 52

    receptacle_border = UI_ACCENT_GREEN if is_flashing else (UI_ACCENT_CYAN if is_incoming else (48, 38, 28))
    receptacle_bg = (14, 28, 20) if is_flashing else ((18, 28, 38) if is_incoming else (16, 10, 8))

    draw_rounded_rect(
        canvas,
        (rx_d, ry_d),
        (rx_d + rw_d, ry_d + rh_d),
        color=receptacle_border,
        thickness=1,
        radius=8,
        fill_color=receptacle_bg,
    )

    # Dashed inner receptacle border
    dash_col = UI_ACCENT_GREEN if is_flashing else (UI_ACCENT_CYAN if is_incoming else (52, 42, 32))
    draw_dashed_rect(
        canvas,
        (rx_d + 1, ry_d + 1),
        (rx_d + rw_d - 1, ry_d + rh_d - 1),
        dash_col,
        dash_len=5,
        gap_len=4,
        thickness=1,
    )

    # 5. Docking Bay State Contents
    cx = rx_d + rw_d // 2
    if is_flashing:
        cv2.putText(
            canvas,
            "SORT COMPLETE",
            (cx - 48, ry_d + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            UI_ACCENT_GREEN,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "[ VERIFIED ]",
            (cx - 38, ry_d + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    elif is_incoming:
        cv2.putText(
            canvas,
            ">> DROP HERE <<",
            (cx - 50, ry_d + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            UI_ACCENT_CYAN,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "RELEASE TO SORT",
            (cx - 50, ry_d + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        # High-tech double downward chevrons
        chevron_col = (130, 145, 165)
        cv2.line(canvas, (cx - 9, ry_d + 18), (cx, ry_d + 26), chevron_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx + 9, ry_d + 18), (cx, ry_d + 26), chevron_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx - 9, ry_d + 27), (cx, ry_d + 35), chevron_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx + 9, ry_d + 27), (cx, ry_d + 35), chevron_col, 2, cv2.LINE_AA)

        # Primary label: Bold white DROP DOCK
        cv2.putText(
            canvas,
            "DROP DOCK",
            (cx - 38, ry_d + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (245, 248, 252),
            2,
            cv2.LINE_AA,
        )
        # Secondary subtitle: Auto Deposit
        cv2.putText(
            canvas,
            "Auto Deposit",
            (cx - 32, ry_d + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (120, 130, 145),
            1,
            cv2.LINE_AA,
        )


# ==============================================================================
# OPENCV MOUSE EVENT CALLBACK
# ==============================================================================
def mouse_callback(event: int, x: int, y: int, flags: int, param: any):
    """Handles click, drag, and drop events for the detected object and JetArm."""
    global DRAG_STATE, ACTIVE_DETECTION, JETARM

    DRAG_STATE.mouse_pos = (x, y)

    # 1. MOUSE BUTTON DOWN
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check HUD Demo Sort Button click
        dbx, dby, dbw, dbh = DEMO_BTN_RECT
        if dbx <= x <= dbx + dbw and dby <= y <= dby + dbh:
            if JETARM is not None and not JETARM.is_running_sequence:
                if ACTIVE_DETECTION is not None:
                    bx, by, bw, bh = ACTIVE_DETECTION.bounding_box
                    JETARM.trigger_sort(
                        (bx + bw // 2, by + bh // 2),
                        ACTIVE_DETECTION.bin_id,
                        ACTIVE_DETECTION.class_name,
                        ACTIVE_DETECTION.confidence,
                        DRAG_STATE.drag_patch,
                    )
                else:
                    JETARM.trigger_sort((320, 350), 2, "Rubiks-Cube", 0.95)
            return

        # Check HUD Auto-Sort Toggle Button click
        abx, aby, abw, abh = AUTOSORT_BTN_RECT
        if abx <= x <= abx + abw and aby <= y <= aby + abh:
            if JETARM is not None:
                JETARM.auto_sort_enabled = not JETARM.auto_sort_enabled
                status_str = "ON" if JETARM.auto_sort_enabled else "OFF"
                print(f"[*] Auto-Sort toggled: {status_str}")
            return

        # Check click inside active detection bounding box to start drag
        if ACTIVE_DETECTION is not None:
            bx, by, bw, bh = ACTIVE_DETECTION.bounding_box
            if bx <= x <= bx + bw and by <= y <= by + bh:
                DRAG_STATE.is_dragging = True
                DRAG_STATE.drag_start_time = time.time()
                DRAG_STATE.original_rect = (bx, by, bw, bh)
                DRAG_STATE.drag_rect = (bx, by, bw, bh)
                DRAG_STATE.drag_offset = (x - bx, y - by)
                DRAG_STATE.drag_class = ACTIVE_DETECTION.class_name
                DRAG_STATE.drag_conf = ACTIVE_DETECTION.confidence
                DRAG_STATE.drag_bin_target = ACTIVE_DETECTION.bin_id

                # Synchronize JetArm: clamp gripper and track dragged box
                if JETARM is not None:
                    JETARM.sync_mouse_drag_start(
                        (bx + bw // 2, by + bh // 2),
                        ACTIVE_DETECTION.class_name,
                        ACTIVE_DETECTION.bin_id,
                    )

    # 2. MOUSE MOVE: Update dragged box coordinates
    elif event == cv2.EVENT_MOUSEMOVE:
        if DRAG_STATE.is_dragging:
            ox, oy = DRAG_STATE.drag_offset
            w, h = DRAG_STATE.drag_rect[2], DRAG_STATE.drag_rect[3]
            new_x = max(0, min(FRAME_WIDTH - w, x - ox))
            new_y = max(0, min(FRAME_HEIGHT - h, y - oy))
            DRAG_STATE.drag_rect = (new_x, new_y, w, h)

            # Synchronize JetArm tip to follow mouse drag center
            if JETARM is not None:
                cur_cx = new_x + w // 2
                cur_cy = new_y + h // 2
                JETARM.sync_mouse_drag_update((cur_cx, cur_cy))

    # 3. MOUSE BUTTON UP: Test drop target
    elif event == cv2.EVENT_LBUTTONUP:
        if DRAG_STATE.is_dragging:
            DRAG_STATE.is_dragging = False
            dx, dy, dw, dh = DRAG_STATE.drag_rect
            drop_cx = dx + dw // 2
            drop_cy = dy + dh // 2

            dropped_bin = None
            for bin_id, bin_info in BIN_ZONES.items():
                bx, by, bw, bh = bin_info["rect"]
                if bx <= drop_cx <= bx + bw and by <= drop_cy <= by + bh:
                    dropped_bin = bin_id
                    break

            # Synchronize JetArm drop release
            if JETARM is not None:
                JETARM.sync_mouse_drag_drop(dropped_bin)

            if dropped_bin is not None:
                duration = time.time() - DRAG_STATE.drag_start_time
                DRAG_STATE.trigger_success(dropped_bin, DRAG_STATE.drag_class)

                print(
                    f"\n>> [SIMULATED SORT] Object '{DRAG_STATE.drag_class.upper()}' "
                    f"({DRAG_STATE.drag_conf * 100:.0f}%) dropped into BIN {dropped_bin} "
                    f"(drag time: {duration:.2f}s)"
                )

                try:
                    sort_logger = param.get("logger") if isinstance(param, dict) else None
                    if sort_logger:
                        sort_logger.log_sort_event(
                            object_class=DRAG_STATE.drag_class,
                            bin_id=dropped_bin,
                            duration_sec=duration,
                            status="SIMULATED",
                            confidence=DRAG_STATE.drag_conf,
                            notes="Mouse-drag visual demo mode",
                        )
                except Exception as err:
                    print(f"   [WARN] Could not write to CSV log: {err}")

            else:
                # Dropped outside any bin zone -> Snap back
                DRAG_STATE.trigger_snap_back()


# ==============================================================================
# DETECTOR INITIALIZATION
# ==============================================================================
def create_detector(conf_thresh: float = 0.50) -> Tuple[BaseDetector, str]:
    """
    Instantiate the appropriate BaseDetector implementation.
    Prioritizes active YOLO model (.pt/.engine) if available;
    falls back cleanly to ColorHSVDetector.
    """
    model_name, model_path = get_active_model_info()

    if model_path and os.path.exists(model_path):
        try:
            detector = YOLODetector(
                model_path=model_path,
                conf_threshold=conf_thresh,
                restrict_to_pickup=True,
            )
            return detector, f"YOLOv8 ({model_name})"
        except Exception as err:
            print(f"[WARN] Failed to load YOLO detector ({err}). Falling back to ColorHSVDetector.")

    # Fallback to HSV
    detector = ColorHSVDetector()
    return detector, "ColorHSVDetector (Baseline)"


# ==============================================================================
# VIDEO CAPTURE HELPER
# ==============================================================================
def open_camera_stream(
    device_idx: int = 0,
    source_type: str = "usb",
    sensor_id: int = 0,
) -> cv2.VideoCapture:
    """
    Opens video capture with cross-platform backend selection:
    - CSI (Jetson): Hardware nvarguscamerasrc via GStreamer (cv2.CAP_GSTREAMER)
    - USB (Linux/Jetson): Video4Linux2 (cv2.CAP_V4L2) with automatic fallback
    - USB (Windows): DirectShow (cv2.CAP_DSHOW) to avoid startup delay
    """
    if str(source_type).lower() == "csi":
        CONFIG.camera.sensor_id = sensor_id
        pipeline = CONFIG.camera.get_gstreamer_pipeline()
        print(f"[*] Opening Jetson CSI Camera via GStreamer (sensor-id={sensor_id}):")
        print(f"    {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[*] CSI Camera stream opened successfully via hardware nvarguscamerasrc.")
            return cap
        print("[!] Warning: GStreamer CSI pipeline failed to open. Falling back to USB camera...")

    # Platform-specific backend for USB camera
    if sys.platform.startswith("win"):
        backend = cv2.CAP_DSHOW
    elif sys.platform.startswith("linux"):
        backend = cv2.CAP_V4L2
    else:
        backend = cv2.CAP_ANY

    cap = cv2.VideoCapture(device_idx, backend)
    # If preferred backend failed on Linux/Windows, fallback to CAP_ANY
    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap = cv2.VideoCapture(device_idx, cv2.CAP_ANY)

    # Fallback to device 0 if non-zero device failed
    if not cap.isOpened() and device_idx != 0:
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap = cv2.VideoCapture(0, cv2.CAP_ANY)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# ==============================================================================
# MAIN LIVE DEMO LOOP
# ==============================================================================
def run_live_demo(
    camera_index: int = 0,
    conf_threshold: float = 0.50,
    image_path: Optional[str] = None,
    source_type: str = "usb",
    sensor_id: int = 0,
):
    """Main application loop rendering the masked pickup zone and drag simulation."""
    global ACTIVE_DETECTION

    print("\n" + "=" * 76)
    print("   AUTONOMOUS OBJECT-SORTING ARM - LIVE DETECTION DEMO MODE")
    print("   (Visualization & Mouse-Drag Pick & Place Simulation Only)")
    print("=" * 76)

    # 1. Initialize detector (BaseDetector interface)
    detector, detector_label = create_detector(conf_thresh=conf_threshold)
    print(f"[*] Active Vision Detector: {detector_label}")
    print(f"[*] Pickup Zone ROI: x={PICKUP_ROI[0]}, y={PICKUP_ROI[1]}, w={PICKUP_ROI[2]}, h={PICKUP_ROI[3]}")
    print("[*] Launching OpenCV Interactive Display Window...\n")

    # 2. Initialize CSV logger
    csv_logger = SortLogger()

    # 3. Setup video source (camera or static image test)
    static_img = None
    cap = None
    if image_path and os.path.exists(image_path):
        static_img = cv2.imread(image_path)
        if static_img is not None:
            static_img = cv2.resize(static_img, (FRAME_WIDTH, FRAME_HEIGHT))
            print(f"[*] Running on static image: {image_path}")

    if static_img is None:
        cap = open_camera_stream(camera_index, source_type=source_type, sensor_id=sensor_id)
        if not cap.isOpened():
            print(f"[!] Warning: Camera {camera_index} (source: {source_type}) could not be opened.")
            print("    Creating synthetic background frame for demonstration.")

    rx, ry, rw, rh = PICKUP_ROI
    rx = max(0, min(rx, FRAME_WIDTH - 1))
    ry = max(0, min(ry, FRAME_HEIGHT - 1))
    rw = max(1, min(rw, FRAME_WIDTH - rx))
    rh = max(1, min(rh, FRAME_HEIGHT - ry))

    # 4. Setup Asynchronous Vision Worker for 25-30+ FPS real-time responsiveness
    vision_worker = AsyncVisionWorker(detector)

    # 5. Setup JetArm Overlay & Kinematics Engine
    # Base is at (320, 466) [highest pixel at y=448, strictly below pickup zone y=440]
    # JetArm grounded to the side of pickup zone at (485, 466)
    # Idle Home diagonal zigzag posture at (485, 225) matching physical hardware photo
    global JETARM
    bin_centers = {
        bin_id: (info["rect"][0] + info["rect"][2] // 2, info["rect"][1] + info["rect"][3] // 2)
        for bin_id, info in BIN_ZONES.items()
    }
    JETARM = JetArmOverlay(
        base_pos=(540, 432),
        home_pos=(540, 210),
        pickup_pos=(rx + rw // 2, ry + rh // 2),
        bin_positions=bin_centers,
        pickup_roi=(rx, ry, rw, rh),
    )

    def on_arm_sort_finished(class_name: str, bin_id: int, duration: float, conf: float):
        print(
            f"\n>> [SIMULATED SORT - HIWONDER JETARM] '{class_name.upper()}' ({conf * 100:.0f}%) "
            f"sorted into BIN {bin_id} (cycle time: {duration:.2f}s)"
        )
        try:
            csv_logger.log_sort_event(
                object_class=class_name,
                bin_id=bin_id,
                duration_sec=duration,
                status="SIMULATED",
                confidence=conf,
                notes="Hiwonder JetArm live camera simulation",
            )
        except Exception as err:
            print(f"   [WARN] Could not write to CSV log: {err}")
        DRAG_STATE.trigger_success(bin_id, class_name)

    def on_arm_bin_flash(bin_id: int):
        DRAG_STATE.flash_bin_id = bin_id
        DRAG_STATE.flash_until = time.time() + 1.8

    JETARM.on_sort_completed = on_arm_sort_finished
    JETARM.on_bin_flash = on_arm_bin_flash

    # 5. Setup OpenCV Window and Mouse Callback
    window_name = "SORT | ARM AI - Live Detection & Hiwonder JetArm Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)
    cv2.setMouseCallback(window_name, mouse_callback, param={"logger": csv_logger})

    # FPS counter metrics
    fps_start = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            # ------------------------------------------------------------------
            # A. Grab raw camera frame
            # ------------------------------------------------------------------
            if static_img is not None:
                frame = static_img.copy()
            elif cap is not None and cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            else:
                # Fallback blank frame if no camera
                frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                cv2.putText(
                    frame, "CAMERA UNAVAILABLE", (170, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2
                )

            frame_count += 1
            now = time.time()
            if now - fps_start >= 1.0:
                fps = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now

            # ------------------------------------------------------------------
            # B. Masking: Render ONLY pickup zone in full color; Dark Navy outside
            # ------------------------------------------------------------------
            detector_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            detector_frame[ry:ry + rh, rx:rx + rw] = frame[ry:ry + rh, rx:rx + rw]

            canvas = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), (20, 12, 8), dtype=np.uint8)

            # Soft reach rings behind arm base
            for rad in (120, 180, 240):
                cv2.ellipse(canvas, (540, 432), (rad, rad), 0, 130, 230, (30, 22, 16), 1, cv2.LINE_AA)

            # Masked camera feed placed cleanly on canvas
            canvas[ry:ry + rh, rx:rx + rw] = frame[ry:ry + rh, rx:rx + rw]

            # Subtle window perimeter glow
            cv2.rectangle(canvas, (2, 2), (FRAME_WIDTH - 3, FRAME_HEIGHT - 3), (45, 35, 25), 1)

            # ------------------------------------------------------------------
            # C. Draw Pickup Zone Frame & Precision Viewfinder Reticle
            # ------------------------------------------------------------------
            # Subtle outer perimeter guide
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (38, 52, 70), 1, cv2.LINE_AA)
            # High-tech camera viewfinder corner brackets
            draw_corner_brackets(canvas, (rx, ry), (rx + rw, ry + rh), UI_ACCENT_CYAN, bracket_len=18, thickness=2)
            # Optical midpoint alignment ticks
            cv2.line(canvas, (rx + rw // 2, ry), (rx + rw // 2, ry + 6), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (rx + rw // 2, ry + rh), (rx + rw // 2, ry + rh - 6), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (rx, ry + rh // 2), (rx + 6, ry + rh // 2), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (rx + rw, ry + rh // 2), (rx + rw - 6, ry + rh // 2), UI_ACCENT_CYAN, 1, cv2.LINE_AA)

            # Center precision optical crosshair reticle when idle
            if ACTIVE_DETECTION is None and not DRAG_STATE.is_dragging:
                pcx = rx + rw // 2
                pcy = ry + rh // 2
                cv2.circle(canvas, (pcx, pcy), 6, UI_ACCENT_CYAN, 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx - 14, pcy), (pcx - 7, pcy), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx + 7, pcy), (pcx + 14, pcy), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx, pcy - 14), (pcx, pcy - 7), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx, pcy + 7), (pcx, pcy + 14), UI_ACCENT_CYAN, 1, cv2.LINE_AA)

            # Dual header plaques positioned above camera zone
            draw_rounded_rect(canvas, (rx, ry - 24), (rx + 118, ry - 5), color=(38, 48, 62), thickness=1, radius=4, fill_color=(18, 24, 34))
            cv2.rectangle(canvas, (rx + 8, ry - 18), (rx + 18, ry - 10), UI_ACCENT_CYAN, 1)
            cv2.circle(canvas, (rx + 13, ry - 14), 2, UI_ACCENT_CYAN, -1)
            cv2.putText(canvas, "AI OPTICAL FEED", (rx + 22, ry - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.30, UI_ACCENT_CYAN, 1, cv2.LINE_AA)

            draw_rounded_rect(canvas, (rx + rw - 76, ry - 24), (rx + rw, ry - 5), color=(35, 42, 54), thickness=1, radius=4, fill_color=(18, 22, 30))
            cv2.putText(canvas, f"ROI ({rw}x{rh})", (rx + rw - 70, ry - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (140, 155, 170), 1, cv2.LINE_AA)

            # ------------------------------------------------------------------
            # D. Draw 3 Polished Bin Zone Cards
            # ------------------------------------------------------------------
            is_flashing = (DRAG_STATE.flash_until > now)
            drag_cx = DRAG_STATE.drag_rect[0] + DRAG_STATE.drag_rect[2] // 2 if DRAG_STATE.is_dragging else -999
            drag_cy = DRAG_STATE.drag_rect[1] + DRAG_STATE.drag_rect[3] // 2 if DRAG_STATE.is_dragging else -999

            for bin_id, bin_info in BIN_ZONES.items():
                bx, by, bw, bh = bin_info["rect"]
                flashing_this_bin = is_flashing and (DRAG_STATE.flash_bin_id == bin_id)
                incoming_this_bin = DRAG_STATE.is_dragging and (bx <= drag_cx <= bx + bw and by <= drag_cy <= by + bh)
                draw_bin_card(canvas, bin_id, bin_info, flashing_this_bin, incoming_this_bin)

            # ------------------------------------------------------------------
            # E. Vision Detection / Interactive Mouse-Drag Handling
            # ------------------------------------------------------------------
            if not DRAG_STATE.is_dragging:
                # Submit isolated camera frame to background worker
                vision_worker.submit_frame(detector_frame)

                det_result = vision_worker.get_result(max_age_sec=1.0)
                if det_result is not None:
                    bx, by, bw, bh = det_result.bounding_box
                    cx = bx + bw // 2
                    cy = by + bh // 2
                    if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
                        det_result = None

                ACTIVE_DETECTION = det_result

                if det_result is not None:
                    bx, by, bw, bh = det_result.bounding_box
                    c_name = det_result.class_name
                    conf = det_result.confidence
                    bin_id = det_result.bin_id

                    box_color = BIN_ZONES.get(bin_id, {}).get("base_color", UI_ACCENT_GREEN)

                    # 1. Bounding box & corner viewfinder brackets
                    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), box_color, 2)
                    draw_corner_brackets(canvas, (bx, by), (bx + bw, by + bh), box_color, bracket_len=8, thickness=2)

                    # 2. Center reticle '+'
                    cx = bx + bw // 2
                    cy = by + bh // 2
                    cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)

                    # 3. Clean floating label badge clamped within pickup zone
                    label_text = f"{c_name.upper()} {int(conf * 100)}% -> BIN {bin_id}"
                    label_y = max(ry + 4, by - 24)
                    draw_hud_badge(
                        canvas,
                        label_text,
                        (bx, label_y),
                        bg_color=(14, 18, 26),
                        text_color=UI_TEXT_TITLE,
                        border_color=box_color,
                        font_scale=0.36,
                        font_thickness=1,
                        padding=(6, 3),
                    )

                    # Store crop patch for drag animation
                    if 0 <= by < by + bh <= FRAME_HEIGHT and 0 <= bx < bx + bw <= FRAME_WIDTH:
                        DRAG_STATE.drag_patch = frame[by:by + bh, bx:bx + bw].copy()

            else:
                dx, dy, dw, dh = DRAG_STATE.drag_rect
                ox, oy, ow, oh = DRAG_STATE.original_rect

                orig_cx = ox + ow // 2
                orig_cy = oy + oh // 2
                cur_cx = dx + dw // 2
                cur_cy = dy + dh // 2

                # 1. Connector guide line
                cv2.line(canvas, (orig_cx, orig_cy), (cur_cx, cur_cy), UI_ACCENT_CYAN, 1, cv2.LINE_AA)

                # 2. Render dragged image patch
                if DRAG_STATE.drag_patch is not None and DRAG_STATE.drag_patch.size > 0:
                    pw, ph = dw, dh
                    patch = cv2.resize(DRAG_STATE.drag_patch, (pw, ph))
                    y1_t, y2_t = max(0, dy), min(FRAME_HEIGHT, dy + ph)
                    x1_t, x2_t = max(0, dx), min(FRAME_WIDTH, dx + pw)
                    patch_cropped = patch[0:(y2_t - y1_t), 0:(x2_t - x1_t)]
                    if patch_cropped.shape[0] > 0 and patch_cropped.shape[1] > 0:
                        canvas[y1_t:y2_t, x1_t:x2_t] = patch_cropped

                # 3. Glowing dragged box & badge
                draw_rounded_rect(canvas, (dx, dy), (dx + dw, dy + dh), UI_ACCENT_CYAN, thickness=2, radius=4)
                drag_label = f"DRAGGING: {DRAG_STATE.drag_class.upper()} ({int(DRAG_STATE.drag_conf * 100)}%)"
                draw_hud_badge(
                    canvas,
                    drag_label,
                    (dx, max(42, dy - 24)),
                    bg_color=(18, 26, 36),
                    text_color=UI_ACCENT_CYAN,
                    border_color=UI_ACCENT_CYAN,
                    font_scale=0.35,
                    padding=(6, 3),
                )

            # ------------------------------------------------------------------
            # G. Hiwonder JetArm Inverse Kinematics & Animated Simulation
            # ------------------------------------------------------------------
            if JETARM is not None:
                # 1. Automatic Sort on Stable Detection (Debounce)
                if ACTIVE_DETECTION is not None and not DRAG_STATE.is_dragging and not JETARM.is_running_sequence:
                    if ACTIVE_DETECTION.class_name == JETARM.debounce_class:
                        JETARM.debounce_hits += 1
                        if JETARM.debounce_hits >= JETARM.debounce_required and JETARM.auto_sort_enabled:
                            abx, aby, abw, abh = ACTIVE_DETECTION.bounding_box
                            JETARM.trigger_sort(
                                (abx + abw // 2, aby + abh // 2),
                                ACTIVE_DETECTION.bin_id,
                                ACTIVE_DETECTION.class_name,
                                ACTIVE_DETECTION.confidence,
                                DRAG_STATE.drag_patch,
                            )
                            JETARM.debounce_hits = 0
                    else:
                        JETARM.debounce_class = ACTIVE_DETECTION.class_name
                        JETARM.debounce_hits = 1
                else:
                    if not DRAG_STATE.is_dragging and not JETARM.is_running_sequence:
                        JETARM.debounce_hits = max(0, JETARM.debounce_hits - 1)

                # 2. Advance Arm Kinematics & Trajectories
                JETARM.update()

                # 3. Draw Hiwonder JetArm, links, joints, shadow, and gripper
                JETARM.draw(canvas)

            # ------------------------------------------------------------------
            # H. Overlay Notification / Confirmation Banner (Middle corridor)
            # ------------------------------------------------------------------
            if DRAG_STATE.status_until > now and DRAG_STATE.status_message:
                banner_w = 460
                banner_h = 28
                bx = (FRAME_WIDTH - banner_w) // 2
                by = 208
                draw_rounded_rect(
                    canvas,
                    (bx, by),
                    (bx + banner_w, by + banner_h),
                    color=DRAG_STATE.status_color,
                    thickness=1,
                    radius=6,
                    fill_color=(14, 20, 28),
                )
                cv2.putText(
                    canvas,
                    DRAG_STATE.status_message,
                    (bx + 14, by + 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    DRAG_STATE.status_color,
                    1,
                    cv2.LINE_AA,
                )

            # ------------------------------------------------------------------
            # I. Top HUD Header Bar & Aligned Interactive Controls
            # ------------------------------------------------------------------
            cv2.rectangle(canvas, (0, 0), (FRAME_WIDTH, 38), (16, 12, 8), -1)
            cv2.line(canvas, (0, 38), (FRAME_WIDTH, 38), (35, 28, 20), 1)

            # Robot logo badge
            draw_rounded_rect(canvas, (12, 6), (44, 32), color=UI_ACCENT_CYAN, thickness=1, radius=5, fill_color=(24, 32, 42))
            cv2.line(canvas, (18, 27), (38, 27), UI_ACCENT_CYAN, 2, cv2.LINE_AA)
            cv2.circle(canvas, (28, 25), 3, UI_ACCENT_CYAN, -1, cv2.LINE_AA)
            cv2.line(canvas, (28, 25), (22, 16), UI_ACCENT_CYAN, 2, cv2.LINE_AA)
            cv2.line(canvas, (22, 16), (32, 11), UI_ACCENT_CYAN, 2, cv2.LINE_AA)
            cv2.line(canvas, (32, 11), (36, 11), UI_ACCENT_CYAN, 2, cv2.LINE_AA)

            # Title: AI VISION (white) ARM (cyan)
            cv2.putText(canvas, "AI VISION", (52, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "ARM", (144, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, UI_ACCENT_CYAN, 2, cv2.LINE_AA)

            # Pill 1: [ ● LIVE ]
            draw_rounded_rect(canvas, (205, 7), (259, 31), color=UI_ACCENT_GREEN, thickness=1, radius=12, fill_color=(18, 48, 28))
            cv2.circle(canvas, (216, 19), 3, UI_ACCENT_GREEN, -1, cv2.LINE_AA)
            cv2.putText(canvas, "LIVE", (224, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.32, UI_ACCENT_GREEN, 1, cv2.LINE_AA)

            # Pill 2: [ 📷 [A] AUTO: ON/OFF ] (Hitbox: AUTOSORT_BTN_RECT = (268, 7, 126, 24))
            auto_active = JETARM.auto_sort_enabled if JETARM else True
            abx, aby, abw, abh = AUTOSORT_BTN_RECT
            if auto_active:
                draw_rounded_rect(canvas, (abx, aby), (abx + abw, aby + abh), color=UI_ACCENT_GREEN, thickness=1, radius=12, fill_color=(24, 44, 34))
                cv2.putText(canvas, "[A] AUTO: ON", (abx + 14, aby + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, UI_ACCENT_GREEN, 1, cv2.LINE_AA)
            else:
                draw_rounded_rect(canvas, (abx, aby), (abx + abw, aby + abh), color=(50, 40, 32), thickness=1, radius=12, fill_color=(22, 16, 12))
                cv2.putText(canvas, "[A] AUTO: OFF", (abx + 12, aby + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, UI_TEXT_MUTED, 1, cv2.LINE_AA)

            # Pill 3: [ ▶ DEMO [D] ] (Hitbox: DEMO_BTN_RECT = (404, 7, 114, 24))
            dbx, dby, dbw, dbh = DEMO_BTN_RECT
            draw_rounded_rect(canvas, (dbx, dby), (dbx + dbw, dby + dbh), color=UI_ACCENT_AMBER, thickness=1, radius=12, fill_color=(28, 38, 50))
            tri_pts = np.array([[dbx + 14, dby + 7], [dbx + 14, dby + 17], [dbx + 22, dby + 12]], np.int32)
            cv2.fillPoly(canvas, [tri_pts], UI_ACCENT_AMBER)
            cv2.putText(canvas, "DEMO [D]", (dbx + 28, dby + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, UI_ACCENT_AMBER, 1, cv2.LINE_AA)

            # Pill 4: [ ⏱ {fps:.1f} FPS ]
            draw_rounded_rect(canvas, (528, 7), (628, 31), color=UI_ACCENT_BLUE, thickness=1, radius=12, fill_color=(28, 22, 16))
            cv2.circle(canvas, (542, 19), 4, UI_ACCENT_BLUE, 1, cv2.LINE_AA)
            cv2.line(canvas, (542, 19), (542, 16), UI_ACCENT_BLUE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{fps:.1f} FPS", (552, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.34, UI_ACCENT_BLUE, 1, cv2.LINE_AA)

            # ------------------------------------------------------------------
            # J. Bottom Console Bar & Telemetry Controls
            # ------------------------------------------------------------------
            cv2.rectangle(canvas, (0, 442), (FRAME_WIDTH, FRAME_HEIGHT), (14, 10, 6), -1)
            cv2.line(canvas, (0, 442), (FRAME_WIDTH, 442), (35, 28, 20), 1)

            # Edit circular button (prominent floating badge matching reference)
            cv2.circle(canvas, (40, 456), 20, (28, 22, 16), -1, cv2.LINE_AA)
            cv2.circle(canvas, (40, 456), 20, (55, 45, 35), 1, cv2.LINE_AA)
            cv2.putText(canvas, "Edit", (26, 461), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)

            # Legend
            cv2.circle(canvas, (76, 461), 4, UI_ACCENT_RED, -1, cv2.LINE_AA)
            cv2.putText(canvas, "Bin 1", (84, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 210, 220), 1, cv2.LINE_AA)

            cv2.circle(canvas, (124, 461), 4, UI_ACCENT_GREEN, -1, cv2.LINE_AA)
            cv2.putText(canvas, "Bin 2", (132, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 210, 220), 1, cv2.LINE_AA)

            cv2.circle(canvas, (172, 461), 4, UI_ACCENT_BLUE, -1, cv2.LINE_AA)
            cv2.putText(canvas, "Bin 3", (180, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 210, 220), 1, cv2.LINE_AA)

            # Tactile Keycaps
            keycaps = [("A", "Auto"), ("Drag", ""), ("Pick", ""), ("Quit", "")]
            kx = 215
            for k_txt, d_txt in keycaps:
                (kw, _), _ = cv2.getTextSize(k_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
                bw_k = kw + 8
                draw_rounded_rect(canvas, (kx, 450), (kx + bw_k, 472), color=(45, 55, 70), thickness=1, radius=3, fill_color=(20, 26, 36))
                cv2.putText(canvas, k_txt, (kx + 4, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.28, UI_ACCENT_CYAN, 1, cv2.LINE_AA)
                kx += bw_k + 4
                if d_txt:
                    cv2.putText(canvas, d_txt, (kx, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130, 145, 160), 1, cv2.LINE_AA)
                    (dw, _), _ = cv2.getTextSize(d_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
                    kx += dw + 6

            # Arm Status Pill
            arm_stat = JETARM.current_status_label if (JETARM and hasattr(JETARM, "current_status_label")) else "IDLE"
            arm_stat_col = JETARM.current_status_color if (JETARM and hasattr(JETARM, "current_status_color")) else UI_ACCENT_GREEN
            arm_pill_bg = (18, 48, 28) if arm_stat == "IDLE" else (28, 38, 50)
            draw_rounded_rect(canvas, (390, 449), (510, 473), color=arm_stat_col, thickness=1, radius=12, fill_color=arm_pill_bg)
            cv2.circle(canvas, (405, 461), 2, arm_stat_col, -1, cv2.LINE_AA)
            cv2.line(canvas, (405, 461), (410, 458), arm_stat_col, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"ARM: {arm_stat}", (416, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.33, arm_stat_col, 1, cv2.LINE_AA)

            # Action button: CYCLES
            cycles_val = JETARM.cycle_count if JETARM else 0
            draw_rounded_rect(canvas, (520, 449), (580, 473), color=(45, 55, 70), thickness=1, radius=10, fill_color=(20, 26, 36))
            cv2.putText(canvas, f"CY: {cycles_val}", (528, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (200, 210, 220), 1, cv2.LINE_AA)

            # Export icon button
            draw_rounded_rect(canvas, (592, 449), (628, 473), color=(45, 55, 70), thickness=1, radius=10, fill_color=(20, 26, 36))
            cv2.line(canvas, (610, 467), (610, 455), (200, 210, 220), 1, cv2.LINE_AA)
            cv2.line(canvas, (607, 458), (610, 455), (200, 210, 220), 1, cv2.LINE_AA)
            cv2.line(canvas, (613, 458), (610, 455), (200, 210, 220), 1, cv2.LINE_AA)

            # Render frame to OpenCV Window
            cv2.imshow(window_name, canvas)

            # Keyboard handler (1ms waitKey)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC to exit
                print("\n[*] Exiting Live Detection Demo...")
                break
            elif key == ord("d") or key == ord("D"):
                # Manual Demo Sort trigger
                if JETARM is not None and not JETARM.is_running_sequence:
                    if ACTIVE_DETECTION is not None:
                        bx, by, bw, bh = ACTIVE_DETECTION.bounding_box
                        JETARM.trigger_sort(
                            (bx + bw // 2, by + bh // 2),
                            ACTIVE_DETECTION.bin_id,
                            ACTIVE_DETECTION.class_name,
                            ACTIVE_DETECTION.confidence,
                            DRAG_STATE.drag_patch,
                        )
                    else:
                        JETARM.trigger_sort((rx + rw // 2, ry + rh // 2), 2, "Rubiks-Cube", 0.95)
            elif key == ord("a") or key == ord("A"):
                # Toggle Auto-Sort on stable detection
                if JETARM is not None:
                    JETARM.auto_sort_enabled = not JETARM.auto_sort_enabled
                    state_msg = "ENABLED" if JETARM.auto_sort_enabled else "DISABLED"
                    print(f"[*] Auto-Sort on detection: {state_msg}")
            elif key == ord("r"):
                # Reset drag feedback
                DRAG_STATE.status_message = ""
                DRAG_STATE.status_until = 0.0

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")

    finally:
        if "vision_worker" in locals() and vision_worker is not None:
            vision_worker.stop()
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        print("[*] Cleanup complete. Window closed.\n")


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomous Object-Sorting Arm - Live Detection & Mouse-Drag Demo Mode"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Confidence threshold for detection (default: 0.50)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["usb", "csi"],
        default="usb",
        help="Camera source: 'usb' (default) or 'csi' (Jetson CSI camera via nvarguscamerasrc GStreamer)",
    )
    parser.add_argument(
        "--sensor-id",
        type=int,
        default=0,
        help="Jetson CSI sensor ID (default: 0 for /dev/nvhost-nvcsi0)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional static image path for testing without live camera",
    )

    args = parser.parse_args()
    run_live_demo(
        camera_index=args.camera,
        conf_threshold=args.conf,
        image_path=args.image,
        source_type=args.source,
        sensor_id=args.sensor_id,
    )
