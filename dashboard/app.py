"""SORT | ARM AI - Web Dashboard Application Backend

Flask-powered high-performance web dashboard server.
Provides:
- MJPEG live video streaming with real-time YOLO bounding box rendering
- Asynchronous sorting pipeline control thread
- Dynamic model uploads (.pt / .engine) and runtime mapping configuration
- Terminal telemetry streaming
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify

# Add jetson module directory to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JETSON_DIR = os.path.join(BASE_DIR, "jetson")
TEST_DIR = os.path.join(BASE_DIR, "test")
sys.path.insert(0, JETSON_DIR)

from config import (
    CONFIG,
    MODELS_DIR,
    MODELS_CONFIG_FILE,
    get_available_models,
    get_active_model_info,
)
from detector import YOLODetector, ColorHSVDetector, DetectionResult
from arm_controller import ArmController
from live_demo import BIN_ZONES, draw_bin_card, draw_dashed_rect
from jetarm_overlay import (
    JetArmOverlay,
    draw_rounded_rect,
    draw_corner_brackets,
    draw_hud_badge,
    UI_BG_DARK,
    UI_PANEL_BG,
    UI_PANEL_HEADER,
    UI_PANEL_BORDER,
    UI_TEXT_TITLE,
    UI_TEXT_BODY,
    UI_TEXT_MUTED,
    UI_ACCENT_CYAN,
    UI_ACCENT_GREEN,
    UI_ACCENT_AMBER,
    UI_ACCENT_RED,
    UI_ACCENT_BLUE,
)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# Global State Container
class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()
        self.preview_running = True  # Automatically active on dashboard launch
        self.pipeline_running = False
        self.mock_arm = True
        self.confidence_threshold = 0.30  # 30% ultra-responsive threshold for live webcam detection
        self.camera_cap = None
        self.camera_index = 0
        self.camera_source = "camera"  # "camera" or "test_image"
        self.detection_mode = "roi"  # Default to "roi" (pickup zone + bins + JetArm simulation)
        self.detector_lock = threading.Lock()
        self.latest_detection: Optional[DetectionResult] = None
        self.latest_detection_time: float = 0.0
        self.flash_bin_id: Optional[int] = None
        self.flash_until: float = 0.0
        self.current_debounce = {"hits": 0, "class": "", "bin": 0, "required": 5}
        self.detector = None
        self.arm = None
        self.jetarm: Optional[JetArmOverlay] = None
        self.logs: List[Dict[str, str]] = []
        self.max_logs = 200

        # Coordinates configuration
        self.coordinates = {
            "pick": {"x": 0, "y": 135, "z": 30},
            "bin1": {"x": 200, "y": 100, "z": 30},
            "bin2": {"x": 0, "y": 150, "z": 30},
            "bin3": {"x": -200, "y": 100, "z": 30},
        }

        # User-uploaded test image (used when no live camera is available)
        self.test_image: Optional[np.ndarray] = None

        # Demo images for synthetic capture fallback
        self.demo_images = []
        self.demo_img_idx = 0
        self.demo_frame_count = 0
        self._load_demo_images()

        # Asynchronous vision worker fields for real-time 30 FPS video streaming
        self.vision_frame: Optional[np.ndarray] = None
        self.vision_event = threading.Event()
        self.vision_infer_ms = 0.0

        # Initialize detector, arm controller, JetArm kinematic overlay, and camera
        self.init_detector()
        self.init_arm()
        self.init_jetarm()
        self.init_camera(self.camera_index)
        self.start_vision_worker()

    def init_jetarm(self):
        """Initialize Hiwonder JetArm 2D Kinematics and visual overlay."""
        rx, ry, rw, rh = CONFIG.vision.roi_pickup_zone
        bin_centers = {
            bin_id: (info["rect"][0] + info["rect"][2] // 2, info["rect"][1] + info["rect"][3] // 2)
            for bin_id, info in BIN_ZONES.items()
        }
        self.jetarm = JetArmOverlay(
            base_pos=(540, 432),
            home_pos=(540, 210),
            pickup_pos=(rx + rw // 2, ry + rh // 2),
            bin_positions=bin_centers,
            pickup_roi=(rx, ry, rw, rh),
        )

    def init_camera(self, cam_idx: int = 0):
        """Initialize or switch camera capture device using DirectShow on Windows."""
        self.camera_index = cam_idx
        if self.camera_cap is not None:
            try:
                self.camera_cap.release()
            except Exception:
                pass
            self.camera_cap = None

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(cam_idx, backend)
        if not cap.isOpened() and cam_idx != 0:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                self.camera_index = 0

        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.camera_cap = cap
            self.camera_source = "camera"
            self.add_log(f"[INFO] Camera {self.camera_index} initialized (DirectShow 640x480).")
        else:
            self.add_log(f"[WARN] Camera {cam_idx} could not be opened. Using fallback frames.")

    def start_vision_worker(self):
        """Dedicated background thread running YOLO/HSV inference without blocking MJPEG stream."""
        def _loop():
            while True:
                self.vision_event.wait(timeout=0.1)
                with self.detector_lock:
                    frame = None if self.vision_frame is None else self.vision_frame.copy()
                    self.vision_frame = None
                    self.vision_event.clear()
                if frame is not None and self.detector is not None:
                    try:
                        t0 = time.time()
                        res = self.detector.detect(frame)
                        with self.detector_lock:
                            self.latest_detection = res
                            self.latest_detection_time = time.time()
                            self.vision_infer_ms = (time.time() - t0) * 1000.0
                    except Exception:
                        pass
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def _load_demo_images(self):
        sample_dir = os.path.join(TEST_DIR, "sample_images")
        files = [
            "rubiks_cube_val.jpg",
            "rubiks_cube_val2.jpg",
            "red_cube_center.jpg",
            "green_cube_center.jpg",
            "blue_cube_center.jpg",
            "empty_tray.jpg",
        ]
        for f in files:
            p = os.path.join(sample_dir, f)
            if os.path.exists(p):
                self.demo_images.append((f, cv2.imread(p)))

    def add_log(self, message: str, level: str = "INFO"):
        with self.lock:
            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
            }
            self.logs.append(entry)
            if len(self.logs) > self.max_logs:
                self.logs.pop(0)

    def init_detector(self):
        active_name, active_path = get_active_model_info()
        if active_path:
            self.detector = YOLODetector(
                model_path=active_path,
                conf_threshold=self.confidence_threshold,
                restrict_to_pickup=(self.detection_mode == "roi"),
            )
            self.add_log(f"Initialized YOLODetector with active model: '{active_name}' (Mode: {self.detection_mode})")
        else:
            self.detector = ColorHSVDetector()
            self.add_log("No active YOLO model found. Initialized ColorHSVDetector fallback.")

    def init_arm(self):
        self.arm = ArmController(simulate=self.mock_arm)
        self.arm.home()
        self.add_log(f"Arm controller initialized (Mock Mode: {self.mock_arm})")


STATE = DashboardState()


# ==============================================================================
# CAMERA & VIDEO STREAMING LOGIC
# ==============================================================================
def get_camera_frame() -> np.ndarray:
    """Fetch live camera frame, user test image, or synthetic demo frame."""
    # 1. Physical live camera if selected as source or active
    if STATE.camera_source == "camera":
        if STATE.camera_cap and STATE.camera_cap.isOpened():
            ret, frame = STATE.camera_cap.read()
            if ret and frame is not None:
                return frame

    # 2. User-uploaded test image or sample cube image
    if STATE.test_image is not None:
        return STATE.test_image.copy()

    # 3. If camera_source is camera but camera_cap was not yet created, try reading if open
    if STATE.camera_cap and STATE.camera_cap.isOpened():
        ret, frame = STATE.camera_cap.read()
        if ret and frame is not None:
            return frame

    # 4. Built-in demo images fallback
    if STATE.demo_images:
        STATE.demo_frame_count += 1
        if STATE.demo_frame_count > 60:
            STATE.demo_frame_count = 0
            STATE.demo_img_idx = (STATE.demo_img_idx + 1) % len(STATE.demo_images)
        _, img = STATE.demo_images[STATE.demo_img_idx]
        jitter = np.random.normal(0, 0.5, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + jitter, 0, 255).astype(np.uint8)

    # 5. Blank canvas
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "NO CAMERA / NO IMAGE", (80, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
    cv2.putText(blank, "Upload a test image or connect a camera", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    return blank


def generate_video_stream():
    """Generator for MJPEG multipart video stream with live detection and Hiwonder JetArm visualization."""
    fps_start = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        if not STATE.preview_running and not STATE.pipeline_running:
            time.sleep(0.1)
            continue

        raw_frame = get_camera_frame()
        if raw_frame is None:
            time.sleep(0.05)
            continue

        # Ensure standard 640x480 resolution
        if raw_frame.shape[0] != 480 or raw_frame.shape[1] != 640:
            raw_frame = cv2.resize(raw_frame, (640, 480))
        h, w = 480, 640

        # FPS calculation
        frame_count += 1
        now = time.time()
        if now - fps_start >= 1.0:
            fps = frame_count / (now - fps_start)
            frame_count = 0
            fps_start = now

        # Run detection to draw live bounding boxes
        result = None
        t_infer = 0.0

        # Non-blocking async inference submission
        if STATE.detector:
            if not STATE.vision_event.is_set():
                with STATE.detector_lock:
                    STATE.vision_frame = raw_frame.copy()
                    STATE.vision_event.set()

            with STATE.detector_lock:
                result = STATE.latest_detection if (time.time() - STATE.latest_detection_time < 0.8) else None
                t_infer = STATE.vision_infer_ms

        if STATE.detection_mode == "roi":
            # 1. Dark matte navy/obsidian background (matching reference design)
            canvas = np.full((h, w, 3), (20, 12, 8), dtype=np.uint8)

            # 2. Soft reach rings behind arm base
            for rad in (120, 180, 240):
                cv2.ellipse(canvas, (540, 432), (rad, rad), 0, 130, 230, (30, 22, 16), 1, cv2.LINE_AA)

            # 3. Live camera feed placed cleanly inside the pickup zone ROI
            rx, ry, rw, rh = CONFIG.vision.roi_pickup_zone
            rx = max(0, min(rx, w - 1))
            ry = max(0, min(ry, h - 1))
            rw = max(1, min(rw, w - rx))
            rh = max(1, min(rh, h - ry))
            canvas[ry:ry + rh, rx:rx + rw] = raw_frame[ry:ry + rh, rx:rx + rw]

            # 4. Subtle window perimeter glow
            cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), (45, 35, 25), 1)

            # 5. Pickup Zone Frame & Precision Viewfinder Reticle
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (38, 52, 70), 1, cv2.LINE_AA)
            draw_corner_brackets(canvas, (rx, ry), (rx + rw, ry + rh), UI_ACCENT_CYAN, bracket_len=18, thickness=2)
            cv2.line(canvas, (rx + rw // 2, ry), (rx + rw // 2, ry + 6), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (rx + rw // 2, ry + rh), (rx + rw // 2, ry + rh - 6), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (rx, ry + rh // 2), (rx + 6, ry + rh // 2), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (rx + rw, ry + rh // 2), (rx + rw - 6, ry + rh // 2), UI_ACCENT_CYAN, 1, cv2.LINE_AA)

            # Center optical reticle when idle
            if result is None and (STATE.jetarm is None or not STATE.jetarm.is_running_sequence):
                pcx = rx + rw // 2
                pcy = ry + rh // 2
                cv2.circle(canvas, (pcx, pcy), 6, UI_ACCENT_CYAN, 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx - 14, pcy), (pcx - 7, pcy), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx + 7, pcy), (pcx + 14, pcy), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx, pcy - 14), (pcx, pcy - 7), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx, pcy + 7), (pcx, pcy + 14), UI_ACCENT_CYAN, 1, cv2.LINE_AA)

            # Dual header plaques above pickup zone
            draw_rounded_rect(canvas, (rx, ry - 24), (rx + 118, ry - 5), color=(38, 48, 62), thickness=1, radius=4, fill_color=(18, 24, 34))
            cv2.rectangle(canvas, (rx + 8, ry - 18), (rx + 18, ry - 10), UI_ACCENT_CYAN, 1)
            cv2.circle(canvas, (rx + 13, ry - 14), 2, UI_ACCENT_CYAN, -1)
            cv2.putText(canvas, "AI OPTICAL FEED", (rx + 22, ry - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.30, UI_ACCENT_CYAN, 1, cv2.LINE_AA)

            draw_rounded_rect(canvas, (rx + rw - 76, ry - 24), (rx + rw, ry - 5), color=(35, 42, 54), thickness=1, radius=4, fill_color=(18, 22, 30))
            cv2.putText(canvas, f"ROI ({rw}x{rh})", (rx + rw - 70, ry - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (140, 155, 170), 1, cv2.LINE_AA)

            # 6. Draw 3 Styled Bin Cards
            is_flashing = (STATE.flash_until > now)
            for bin_id, bin_info in BIN_ZONES.items():
                flashing_this_bin = is_flashing and (STATE.flash_bin_id == bin_id)
                draw_bin_card(canvas, bin_id, bin_info, flashing_this_bin, is_incoming=False)

            # 7. Draw Detected Object in Pickup Zone
            if result:
                bx, by, bw, bh = result.bounding_box
                cx, cy = result.center
                if rx - 20 <= cx <= rx + rw + 20 and ry - 20 <= cy <= ry + rh + 20:
                    box_color = BIN_ZONES.get(result.bin_id, {}).get("base_color", UI_ACCENT_GREEN)
                    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), box_color, 2)
                    draw_corner_brackets(canvas, (bx, by), (bx + bw, by + bh), box_color, bracket_len=8, thickness=2)
                    cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)

                    label_text = f"{result.class_name.upper()} {int(result.confidence * 100)}% -> BIN {result.bin_id}"
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

                    # Auto-sort on stable detection if arm is idle and auto_sort_enabled is True
                    if STATE.jetarm and STATE.jetarm.auto_sort_enabled and not STATE.jetarm.is_running_sequence and not STATE.pipeline_running:
                        if result.class_name == STATE.jetarm.debounce_class:
                            STATE.jetarm.debounce_hits += 1
                            if STATE.jetarm.debounce_hits >= 12:
                                patch = raw_frame[by:by + bh, bx:bx + bw].copy() if 0 <= by < by + bh <= h and 0 <= bx < bx + bw <= w else None
                                STATE.jetarm.trigger_sort((cx, cy), result.bin_id, result.class_name, result.confidence, patch)
                                STATE.flash_bin_id = result.bin_id
                                STATE.flash_until = now + 1.8
                                STATE.jetarm.debounce_hits = 0
                                STATE.add_log(f"[AUTO-SORT] Detected '{result.class_name.upper()}' -> Auto-Sorting to Bin {result.bin_id}", level="SUCCESS")
                        else:
                            STATE.jetarm.debounce_class = result.class_name
                            STATE.jetarm.debounce_hits = 1
                    elif STATE.jetarm and not STATE.jetarm.is_running_sequence:
                        STATE.jetarm.debounce_hits = max(0, STATE.jetarm.debounce_hits - 1)

            # 8. Advance Arm Kinematics & Draw Hiwonder JetArm
            if STATE.jetarm:
                STATE.jetarm.update()
                STATE.jetarm.draw(canvas)

            annotated = canvas

        else:
            # Full frame mode
            if result and hasattr(result, "annotated_frame") and result.annotated_frame is not None:
                annotated = result.annotated_frame
            elif hasattr(STATE.detector, "last_annotated_frame") and STATE.detector.last_annotated_frame is not None:
                annotated = STATE.detector.last_annotated_frame
            else:
                annotated = raw_frame.copy()

        # Top HUD Bar
        cv2.rectangle(annotated, (0, 0), (w, 36), (16, 12, 8), -1)
        cv2.line(annotated, (0, 36), (w, 36), (35, 28, 20), 1)

        # Robot logo badge
        draw_rounded_rect(annotated, (12, 5), (42, 31), color=UI_ACCENT_CYAN, thickness=1, radius=5, fill_color=(24, 32, 42))
        cv2.line(annotated, (18, 26), (36, 26), UI_ACCENT_CYAN, 2, cv2.LINE_AA)
        cv2.circle(annotated, (27, 24), 3, UI_ACCENT_CYAN, -1, cv2.LINE_AA)
        cv2.line(annotated, (27, 24), (22, 15), UI_ACCENT_CYAN, 2, cv2.LINE_AA)
        cv2.line(annotated, (22, 15), (32, 10), UI_ACCENT_CYAN, 2, cv2.LINE_AA)

        # Title: AI VISION (white) ARM (cyan)
        cv2.putText(annotated, "AI VISION", (48, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(annotated, "ARM", (130, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, UI_ACCENT_CYAN, 2, cv2.LINE_AA)

        # Pill 1: [ ● AUTO ] or [ ● MANUAL ] or [ ● SORTING ]
        if STATE.pipeline_running:
            status_text = "SORTING"
            status_color = UI_ACCENT_GREEN
        elif STATE.jetarm and STATE.jetarm.auto_sort_enabled:
            status_text = "AUTO"
            status_color = UI_ACCENT_GREEN
        else:
            status_text = "MANUAL"
            status_color = UI_ACCENT_CYAN
        draw_rounded_rect(annotated, (185, 6), (268, 30), color=status_color, thickness=1, radius=12, fill_color=(18, 42, 24) if status_color == UI_ACCENT_GREEN else (18, 28, 42))
        cv2.circle(annotated, (198, 18), 3, status_color, -1, cv2.LINE_AA)
        cv2.putText(annotated, status_text, (206, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

        # Pill 2: [ MODEL ]
        active_name, _ = get_active_model_info()
        mod_label = (active_name or "YOLOv8").replace(".pt", "").replace(".engine", "")[:12]
        draw_rounded_rect(annotated, (262, 6), (360, 30), color=(50, 60, 75), thickness=1, radius=12, fill_color=(20, 26, 36))
        cv2.putText(annotated, mod_label, (272, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, UI_ACCENT_CYAN, 1, cv2.LINE_AA)

        # Pill 3: FPS & Infer time
        draw_rounded_rect(annotated, (w - 140, 6), (w - 12, 30), color=(50, 60, 75), thickness=1, radius=12, fill_color=(20, 26, 36))
        cv2.putText(annotated, f"{fps:.1f} FPS | {t_infer:.0f}ms", (w - 132, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 215, 230), 1, cv2.LINE_AA)

        # Debounce progress bar on HUD if tracking candidate
        if STATE.current_debounce["hits"] > 0:
            hits = STATE.current_debounce["hits"]
            req = STATE.current_debounce["required"]
            cls_n = STATE.current_debounce["class"].upper()
            bar_w = 170
            bar_h = 10
            bx, by = (w - bar_w) // 2, h - 25
            cv2.rectangle(annotated, (bx - 10, by - 16), (bx + bar_w + 10, by + bar_h + 8), (6, 14, 24), -1)
            cv2.rectangle(annotated, (bx, by), (bx + bar_w, by + bar_h), (50, 70, 90), 1)
            fill_w = int(bar_w * min(1.0, hits / req))
            lock_col = (0, 255, 0) if hits >= req else (0, 212, 255)
            cv2.rectangle(annotated, (bx, by), (bx + fill_w, by + bar_h), lock_col, -1)
            cv2.putText(annotated, f"LOCKING: {cls_n} ({hits}/{req})", (bx, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, lock_col, 1, cv2.LINE_AA)

        # Encode frame as JPEG
        ret, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )
        time.sleep(0.025)


# ==============================================================================
# AUTONOMOUS SORTING WORKER THREAD
# ==============================================================================
def sorting_worker():
    """Background thread running the autonomous sorting pipeline."""
    STATE.add_log("[START] Autonomous Object-Sorting Pipeline started.")
    consecutive_hits = 0
    candidate_class = None
    candidate_bin = None

    while STATE.pipeline_running:
        result = None
        # Reuse instantaneous detection from live video stream to avoid duplicate inference and camera contention
        if time.time() - STATE.latest_detection_time < 0.25:
            result = STATE.latest_detection
        else:
            frame = get_camera_frame()
            if frame is not None and STATE.detector:
                try:
                    with STATE.detector_lock:
                        result = STATE.detector.detect(frame)
                        STATE.latest_detection = result
                        STATE.latest_detection_time = time.time()
                except Exception as e:
                    STATE.add_log(f"[ERROR] Detection exception: {e}", level="ERROR")

        if result:
            if result.class_name == candidate_class:
                consecutive_hits += 1
            else:
                candidate_class = result.class_name
                candidate_bin = result.bin_id
                consecutive_hits = 1

            STATE.current_debounce = {
                "hits": consecutive_hits,
                "class": candidate_class or "",
                "bin": candidate_bin or 0,
                "required": CONFIG.debounce.required_consecutive_detections,
            }

            if consecutive_hits >= CONFIG.debounce.required_consecutive_detections:
                STATE.add_log(
                    f"[CONFIRMED] Detected '{candidate_class.upper()}' (Bin {candidate_bin}). Triggering Arm Cycle!"
                )
                
                # Execute 10-step arm movement
                t0 = time.time()
                success = STATE.arm.move_to_bin(candidate_bin)
                dt = time.time() - t0

                if success:
                    STATE.add_log(
                        f"[SUCCESS] Sorted '{candidate_class}' into Bin {candidate_bin} in {dt:.2f}s",
                        level="SUCCESS",
                    )
                else:
                    STATE.add_log(f"[ERROR] Arm cycle failed for Bin {candidate_bin}", level="ERROR")

                # Settle cooldown
                consecutive_hits = 0
                candidate_class = None
                STATE.current_debounce = {"hits": 0, "class": "", "bin": 0, "required": CONFIG.debounce.required_consecutive_detections}
                time.sleep(CONFIG.debounce.post_sort_cooldown_sec)
        else:
            consecutive_hits = 0
            candidate_class = None
            STATE.current_debounce = {"hits": 0, "class": "", "bin": 0, "required": CONFIG.debounce.required_consecutive_detections}

        time.sleep(0.05)

    STATE.add_log("[STOP] Sorting pipeline stopped. Arm parked safely at HOME.")
    try:
        STATE.arm.home()
    except Exception:
        pass


# ==============================================================================
# FLASK WEB ROUTES & REST APIS
# ==============================================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status", methods=["GET"])
def get_status():
    active_name, _ = get_active_model_info()
    arm_state = STATE.jetarm.state if STATE.jetarm else "IDLE"
    auto_sort = STATE.jetarm.auto_sort_enabled if STATE.jetarm else True
    return jsonify({
        "preview_running": STATE.preview_running,
        "pipeline_running": STATE.pipeline_running,
        "mock_arm": STATE.mock_arm,
        "confidence": STATE.confidence_threshold,
        "active_model": active_name or "None",
        "detection_mode": STATE.detection_mode,
        "camera_source": STATE.camera_source,
        "camera_index": STATE.camera_index,
        "arm_state": arm_state,
        "sort_mode": "auto" if auto_sort else "manual",
        "auto_sort": auto_sort,
    })


@app.route("/api/arm/mode", methods=["GET", "POST"])
def handle_arm_mode():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "auto")  # "auto" or "manual"
        if STATE.jetarm:
            STATE.jetarm.auto_sort_enabled = (mode == "auto")
        STATE.add_log(f"[CONFIG] Live Cam Sorting Mode set to: '{mode.upper()}'")
        return jsonify({"success": True, "mode": mode, "auto_sort": STATE.jetarm.auto_sort_enabled if STATE.jetarm else True})
    return jsonify({
        "mode": "auto" if (STATE.jetarm and STATE.jetarm.auto_sort_enabled) else "manual",
        "auto_sort": STATE.jetarm.auto_sort_enabled if STATE.jetarm else True,
    })


@app.route("/api/preview/start", methods=["POST"])
def start_preview():
    data = request.get_json(silent=True) or {}
    cam_idx = int(data.get("camera_index", STATE.camera_index))
    STATE.preview_running = True
    STATE.init_camera(cam_idx)
    return jsonify({"success": True, "camera_index": STATE.camera_index})


@app.route("/api/preview/stop", methods=["POST"])
def stop_preview():
    STATE.preview_running = False
    if STATE.camera_cap and STATE.camera_cap.isOpened():
        STATE.camera_cap.release()
        STATE.camera_cap = None
        STATE.add_log("[INFO] Camera released.")
    else:
        STATE.add_log("[INFO] Preview stopped.")
    return jsonify({"success": True})


@app.route("/api/cameras", methods=["GET"])
def get_cameras():
    """Scan and list available camera devices."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cameras = []
    for idx in range(3):
        if STATE.camera_cap and STATE.camera_cap.isOpened() and STATE.camera_index == idx:
            cameras.append({"index": idx, "name": f"Camera {idx} (Active)", "active": True})
            continue
        test_cap = cv2.VideoCapture(idx, backend)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                cameras.append({"index": idx, "name": f"Camera {idx}", "active": False})
            test_cap.release()
    return jsonify({"cameras": cameras, "current_index": STATE.camera_index, "camera_source": STATE.camera_source})


@app.route("/api/detection/mode", methods=["GET", "POST"])
def handle_detection_mode():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        mode = data.get("mode", "full_frame")  # "full_frame" or "roi"
        STATE.detection_mode = mode
        if STATE.detector and hasattr(STATE.detector, "restrict_to_pickup"):
            STATE.detector.restrict_to_pickup = (mode == "roi")
        STATE.add_log(f"[CONFIG] Detection mode switched to: '{mode.upper()}'")
        return jsonify({"success": True, "mode": STATE.detection_mode})
    return jsonify({
        "mode": STATE.detection_mode,
        "restrict_to_pickup": getattr(STATE.detector, "restrict_to_pickup", False) if STATE.detector else False
    })


@app.route("/api/detection/current", methods=["GET"])
def get_current_detection():
    det = STATE.latest_detection if (time.time() - STATE.latest_detection_time < 0.6) else None
    det_dict = None
    if det:
        det_dict = {
            "class_name": det.class_name,
            "confidence": float(det.confidence),
            "bin_id": int(det.bin_id),
            "bounding_box": [int(v) for v in det.bounding_box],
            "center": [int(v) for v in det.center],
        }
    return jsonify({
        "detection": det_dict,
        "pickup_roi": list(CONFIG.vision.roi_pickup_zone),
        "bin_zones": {k: {"name": v["name"], "rect": list(v["rect"])} for k, v in BIN_ZONES.items()},
        "mode": STATE.detection_mode,
        "arm_state": STATE.jetarm.state if STATE.jetarm else "IDLE",
    })


@app.route("/api/simulated_sort", methods=["POST"])
def simulated_sort():
    data = request.get_json(silent=True) or {}
    class_name = data.get("class_name", "object")
    bin_id = int(data.get("bin_id", 1))
    conf = float(data.get("confidence", 0.95))

    STATE.flash_bin_id = bin_id
    STATE.flash_until = time.time() + 1.8
    STATE.add_log(
        f"[SIMULATED SORT] Detected '{class_name.upper()}' ({conf * 100:.0f}%) manually sorted to BIN {bin_id} via Interactive Demo.",
        level="SUCCESS",
    )

    if STATE.jetarm is not None:
        rx, ry, rw, rh = CONFIG.vision.roi_pickup_zone
        center = (rx + rw // 2, ry + rh // 2)
        if STATE.latest_detection and (time.time() - STATE.latest_detection_time < 1.0):
            center = STATE.latest_detection.center
        STATE.jetarm.trigger_sort(
            pickup_pos=center,
            bin_id=bin_id,
            class_name=class_name,
            conf=conf,
        )

    try:
        from logger import SortLogger
        SortLogger().log_sort_event(
            object_class=class_name,
            bin_id=bin_id,
            duration_sec=1.5,
            status="SIMULATED",
            confidence=conf,
            notes="Dashboard interactive mouse-drag simulation",
        )
    except Exception as err:
        pass

    return jsonify({"success": True, "bin_id": bin_id, "class_name": class_name})


@app.route("/api/preview/switch_to_camera", methods=["POST"])
def switch_to_camera():
    STATE.camera_source = "camera"
    STATE.add_log("[CONFIG] Switched video source to Live Camera.")
    return jsonify({"success": True, "camera_source": "camera"})


@app.route("/api/preview/upload_image", methods=["POST"])
def upload_test_image():
    """Upload a test image (.jpg/.png) — YOLO model will run inference on it."""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file in request"}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    allowed = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        return jsonify({"success": False, "error": "Unsupported image format"}), 400

    img_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "Could not decode image"}), 400

    # Resize to 640x480 for consistency
    img = cv2.resize(img, (640, 480))
    STATE.test_image = img
    STATE.camera_source = "test_image"
    if hasattr(STATE.detector, "restrict_to_pickup"):
        STATE.detector.restrict_to_pickup = False
    STATE.preview_running = True  # Auto-start preview
    STATE.add_log(f"[INFO] Test image '{file.filename}' loaded ({img.shape[1]}x{img.shape[0]}). Model running on it.")
    return jsonify({"success": True, "filename": file.filename})


@app.route("/api/preview/load_sample", methods=["POST"])
def load_sample_image():
    """Load default sample image of Rubik's cube for quick testing."""
    sample_path = os.path.join(TEST_DIR, "sample_images", "rubiks_cube_val.jpg")
    if not os.path.exists(sample_path):
        return jsonify({"success": False, "error": "Sample image not found"}), 404

    img = cv2.imread(sample_path)
    if img is None:
        return jsonify({"success": False, "error": "Could not read sample image"}), 500

    img = cv2.resize(img, (640, 480))
    STATE.test_image = img
    STATE.camera_source = "test_image"
    if hasattr(STATE.detector, "restrict_to_pickup"):
        STATE.detector.restrict_to_pickup = False
    STATE.preview_running = True
    STATE.add_log("[INFO] Loaded sample Rubik's Cube image. Model running live inference.")
    return jsonify({"success": True, "filename": "rubiks_cube_val.jpg"})



@app.route("/api/confidence", methods=["POST"])
def set_confidence():
    data = request.get_json() or {}
    val = float(data.get("confidence", 0.75))
    STATE.confidence_threshold = max(0.05, min(0.99, val))
    if isinstance(STATE.detector, YOLODetector):
        STATE.detector.conf_threshold = STATE.confidence_threshold
    STATE.add_log(f"[CONFIG] Confidence threshold set to {STATE.confidence_threshold*100:.0f}%")
    return jsonify({"success": True, "confidence": STATE.confidence_threshold})


@app.route("/api/models", methods=["GET"])
def get_models():
    models_list = []
    active_name, _ = get_active_model_info()
    available = get_available_models()

    for m_file in available:
        full_p = os.path.join(MODELS_DIR, m_file)
        size_mb = os.path.getsize(full_p) / (1024.0 * 1024.0)
        stem = os.path.splitext(m_file)[0]
        map_p = os.path.join(MODELS_DIR, f"{stem}_mapping.json")

        mapping_data = {}
        if os.path.exists(map_p):
            try:
                with open(map_p, "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)
            except Exception:
                pass

        classes = list(mapping_data.keys())
        if not classes:
            classes = ["class_0", "class_1", "class_2"]

        models_list.append({
            "filename": m_file,
            "size": f"{size_mb:.1f} MB",
            "classes": classes,
            "mapping": mapping_data,
        })

    return jsonify({
        "active_model": active_name,
        "models": models_list,
    })


@app.route("/api/models/upload", methods=["POST"])
def upload_model():
    if "model" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["model"]
    if not file.filename:
        return jsonify({"success": False, "error": "Empty filename"}), 400

    if not (file.filename.endswith(".pt") or file.filename.endswith(".engine")):
        return jsonify({"success": False, "error": "Invalid file type. Only .pt and .engine supported."}), 400

    os.makedirs(MODELS_DIR, exist_ok=True)
    target_path = os.path.join(MODELS_DIR, file.filename)
    file.save(target_path)
    STATE.add_log(f"[UPLOAD] Model '{file.filename}' uploaded successfully.")

    # Automatically set as active model
    cfg_data = {
        "active_model": file.filename,
        "selected_at": datetime.now().isoformat(),
        "model_path": target_path,
    }
    with open(MODELS_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)

    STATE.init_detector()
    return jsonify({"success": True, "filename": file.filename})


@app.route("/api/models/select", methods=["POST"])
def select_model():
    data = request.get_json() or {}
    model_name = data.get("model")
    available = get_available_models()

    if model_name not in available:
        return jsonify({"success": False, "error": "Model not found"}), 404

    cfg_data = {
        "active_model": model_name,
        "selected_at": datetime.now().isoformat(),
        "model_path": os.path.join(MODELS_DIR, model_name),
    }
    with open(MODELS_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)

    STATE.init_detector()
    STATE.add_log(f"[CONFIG] Switched active model to '{model_name}'")
    return jsonify({"success": True, "active_model": model_name})


@app.route("/api/models/mapping", methods=["POST"])
def update_mapping():
    data = request.get_json() or {}
    model_name = data.get("model")
    new_mapping = data.get("mapping", {})

    if not model_name:
        return jsonify({"success": False, "error": "Model name required"}), 400

    stem = os.path.splitext(model_name)[0]
    map_p = os.path.join(MODELS_DIR, f"{stem}_mapping.json")

    with open(map_p, "w", encoding="utf-8") as f:
        json.dump(new_mapping, f, indent=2)

    if isinstance(STATE.detector, YOLODetector):
        STATE.detector.reload_mapping()

    STATE.add_log(f"[CONFIG] Updated class-to-bin mapping for '{model_name}'")
    return jsonify({"success": True})


@app.route("/api/pipeline/start", methods=["POST"])
def start_pipeline():
    if STATE.pipeline_running:
        return jsonify({"success": True, "message": "Already running"})

    STATE.pipeline_running = True
    STATE.preview_running = True
    t = threading.Thread(target=sorting_worker, daemon=True)
    t.start()
    return jsonify({"success": True})


@app.route("/api/pipeline/stop", methods=["POST"])
def stop_pipeline():
    STATE.pipeline_running = False
    return jsonify({"success": True})


@app.route("/api/pipeline/toggle_mock", methods=["POST"])
def toggle_mock():
    data = request.get_json() or {}
    STATE.mock_arm = bool(data.get("mock", True))
    STATE.init_arm()
    return jsonify({"success": True, "mock_arm": STATE.mock_arm})


@app.route("/api/arm/coordinates", methods=["GET", "POST"])
def handle_coordinates():
    if request.method == "POST":
        data = request.get_json() or {}
        if "coordinates" in data:
            STATE.coordinates = data["coordinates"]
            STATE.add_log("[CONFIG] Arm coordinates updated from dashboard.")
        return jsonify({"success": True, "coordinates": STATE.coordinates})
    return jsonify({"success": True, "coordinates": STATE.coordinates})


@app.route("/api/logs", methods=["GET"])
def get_logs():
    since = request.args.get("since", "")
    with STATE.lock:
        if since:
            new_logs = [l for l in STATE.logs if l["timestamp"] > since]
        else:
            new_logs = list(STATE.logs)

    return jsonify({
        "logs": new_logs,
        "pipeline_running": STATE.pipeline_running,
        "mock_arm": STATE.mock_arm,
    })


@app.route("/api/logs/clear", methods=["POST"])
def clear_logs():
    with STATE.lock:
        STATE.logs.clear()
    return jsonify({"success": True})


def create_app():
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f">> Starting SORT | ARM AI Dashboard on http://localhost:{port}...")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
