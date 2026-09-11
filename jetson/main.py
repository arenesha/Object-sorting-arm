"""Autonomous Object-Sorting Robotic Arm - Main Orchestrator

Single-board hardware + AI system running entirely on NVIDIA Jetson Orin Nano 8GB.
Executes the closed-loop cycle:
  Camera Capture -> OpenCV HSV Detection -> Debounce State Machine ->
  PCA9685 Direct Servo Arm Pick & Place -> Event Logging -> Resume Detection

Usage:
  python main.py                     # Run live system on Jetson with USB camera
  python main.py --csi               # Run live with Jetson CSI camera (nvarguscamerasrc)
  python main.py --simulate          # Run with arm in simulation mode
  python main.py --headless          # Run without GUI window (headless/daemon mode)
  python main.py --demo              # Run live synthetic demo feeding sample images
"""

import os
import sys
import time
import signal
import argparse
from typing import Optional, Dict
import cv2
import numpy as np

from config import CONFIG, BIN_CONFIG, get_active_model_info
from detector import ColorHSVDetector, YOLODetector, DetectionResult
from arm_controller import ArmController
from logger import SortLogger


class SortingStateMachine:
    """State constants for the orchestrator."""
    IDLE = "IDLE (SCANNING)"
    DEBOUNCING = "DEBOUNCING"
    SORTING = "SORTING (ARM ACTIVE)"
    COOLDOWN = "COOLDOWN"


class AutonomousSortingSystem:
    """Master orchestrator integrating Vision, State Machine, and Arm Control."""

    def __init__(
        self,
        simulate_arm: bool = False,
        camera_source: str = "usb",
        camera_index: int = 0,
        headless: bool = False,
        demo_mode: bool = False,
        detector_type: str = "auto",
        model_path: Optional[str] = None,
    ):
        self.headless = headless
        self.demo_mode = demo_mode
        self.running = True

        # Initialize telemetry logger
        self.logger = SortLogger()
        self.logger.info("Initializing Autonomous Object-Sorting System on Jetson Orin Nano...")

        # Initialize Arm Controller
        self.arm = ArmController(simulate=simulate_arm)
        self.arm.home()

        # Initialize Vision Detector (Dynamic YOLO or HSV)
        active_model_name, active_model_path = get_active_model_info()
        target_model = model_path or active_model_path

        if detector_type == "yolo" or (detector_type == "auto" and target_model is not None):
            self.detector_name = "YOLO"
            self.logger.info(f"Using Dynamic YOLODetector (Active Model: {active_model_name or 'Simulated'})")
            self.detector = YOLODetector(model_path=target_model)
        else:
            self.detector_name = "HSV"
            self.logger.info("Using baseline ColorHSVDetector")
            self.detector = ColorHSVDetector()

        # Camera setup
        self.camera_source = camera_source
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.demo_images = []
        self.demo_img_idx = 0
        self.demo_frame_count = 0

        self._init_camera()

        # State Machine Tracking
        self.state = SortingStateMachine.IDLE
        self.candidate_class: Optional[str] = None
        self.candidate_bin: Optional[int] = None
        self.consecutive_count: int = 0
        self.empty_count: int = 0
        self.cooldown_end_time: float = 0.0

        # Session Metrics
        self.sort_tally: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self.last_sorted_info = "None"

        # Register OS signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _init_camera(self):
        """Initialize camera stream or demo image pipeline."""
        if self.demo_mode:
            self.logger.info("Running in DEMO MODE: Synthesizing real-time camera frames.")
            sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test", "sample_images"))
            files = ["red_cube_center.jpg", "green_cube_center.jpg", "blue_cube_center.jpg", "empty_tray.jpg"]
            for f in files:
                p = os.path.join(sample_dir, f)
                if os.path.exists(p):
                    self.demo_images.append((f, cv2.imread(p)))
            if not self.demo_images:
                self.logger.warning("No demo images found. Please run test/generate_sample_data.py first.")
            return

        if self.camera_source == "csi":
            pipeline = CONFIG.camera.get_gstreamer_pipeline()
            self.logger.info(f"Opening Jetson CSI Camera via GStreamer: {pipeline}")
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.logger.info(f"Opening USB Camera on device index {self.camera_index}...")
            backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else (cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
            self.cap = cv2.VideoCapture(self.camera_index, backend)
            if not self.cap.isOpened() and backend != cv2.CAP_ANY:
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_ANY)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.camera.frame_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.camera.frame_height)
                self.cap.set(cv2.CAP_PROP_FPS, CONFIG.camera.fps)

        if not self.cap or not self.cap.isOpened():
            self.logger.warning(
                f"Camera device could not be opened (source: {self.camera_source}, index: {self.camera_index}). "
                "Switching automatically to synthetic DEMO stream."
            )
            self.demo_mode = True
            self._init_camera()

    def _signal_handler(self, sig, frame):
        """Gracefully handle Ctrl+C / SIGINT by homing the arm and releasing resources."""
        self.logger.info("Shutdown signal received. Performing graceful park...")
        self.running = False

    def get_next_frame(self) -> Optional[np.ndarray]:
        """Fetch the next video frame from camera hardware or demo generator."""
        if self.demo_mode:
            if not self.demo_images:
                return None
            # Cycle through sample images every ~60 frames
            self.demo_frame_count += 1
            if self.demo_frame_count > 70:
                self.demo_frame_count = 0
                self.demo_img_idx = (self.demo_img_idx + 1) % len(self.demo_images)
            _, img = self.demo_images[self.demo_img_idx]
            # Add subtle jitter noise to simulate live optical stream
            jitter = np.random.normal(0, 1.2, img.shape).astype(np.int16)
            noisy_frame = np.clip(img.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
            time.sleep(0.03)  # Emulate 30 FPS
            return noisy_frame

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                self.logger.warning("Camera frame read failed.")
                return None
        return None

    def draw_hud(self, frame: np.ndarray, result: Optional[DetectionResult]) -> np.ndarray:
        """Render comprehensive Heads-Up Display (HUD) overlay on frame."""
        hud = frame.copy()
        h, w = hud.shape[:2]

        # 1. Status Bar Header
        header_h = 44
        cv2.rectangle(hud, (0, 0), (w, header_h), (25, 25, 30), -1)
        cv2.line(hud, (0, header_h), (w, header_h), (80, 80, 90), 1)

        # System State tag with color code
        state_colors = {
            SortingStateMachine.IDLE: (200, 200, 200),
            SortingStateMachine.DEBOUNCING: (0, 220, 255),
            SortingStateMachine.SORTING: (0, 255, 0),
            SortingStateMachine.COOLDOWN: (255, 140, 0),
        }
        tag_color = state_colors.get(self.state, (255, 255, 255))
        cv2.putText(
            hud,
            f"STATE: {self.state}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            tag_color,
            2,
            cv2.LINE_AA,
        )

        # Hardware & Detector mode indicator
        hw_tag = "[SIMULATED ARM]" if self.arm.simulate else "[PCA9685 I2C]"
        engine_tag = f"[{self.detector_name}]"
        cv2.putText(
            hud,
            f"{engine_tag} {hw_tag}",
            (w - 240, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 200) if not self.arm.simulate else (100, 180, 255),
            1,
            cv2.LINE_AA,
        )

        # 2. Debounce Progress Meter (shown during DEBOUNCING state)
        if self.state == SortingStateMachine.DEBOUNCING:
            req = CONFIG.debounce.required_consecutive_detections
            pct = min(1.0, self.consecutive_count / req)
            bar_x, bar_y, bar_w, bar_h = 15, 55, 180, 16
            cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            cv2.rectangle(
                hud,
                (bar_x, bar_y),
                (bar_x + int(bar_w * pct), bar_y + bar_h),
                (0, 220, 255),
                -1,
            )
            cv2.rectangle(hud, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
            cv2.putText(
                hud,
                f"Locking: {self.consecutive_count}/{req}",
                (bar_x + bar_w + 10, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 255),
                1,
                cv2.LINE_AA,
            )

        # 3. Bottom Stats Overlay
        footer_y = h - 35
        cv2.rectangle(hud, (0, footer_y - 10), (w, h), (20, 20, 25), -1)
        stats_str = (
            f"TALLY: Bin 1 (Red): {self.sort_tally[1]} | "
            f"Bin 2 (Green): {self.sort_tally[2]} | "
            f"Bin 3 (Blue): {self.sort_tally[3]} | "
            f"Last: {self.last_sorted_info}"
        )
        cv2.putText(
            hud,
            stats_str,
            (12, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 225),
            1,
            cv2.LINE_AA,
        )

        return hud

    def run(self):
        """Main real-time closed loop."""
        self.logger.info("=== Autonomous Object-Sorting System Active ===")
        self.logger.info("Watching pickup zone. Press 'q' in OpenCV window or Ctrl+C to stop.")

        window_name = "Jetson Orin Nano - Object Sorting Arm"
        if not self.headless:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while self.running:
                frame = self.get_next_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue

                now = time.time()

                # -------------------------------------------------------------
                # State 1: COOLDOWN (waiting after completed sort)
                # -------------------------------------------------------------
                if self.state == SortingStateMachine.COOLDOWN:
                    if now >= self.cooldown_end_time:
                        self.state = SortingStateMachine.IDLE
                        self.candidate_class = None
                        self.consecutive_count = 0
                        self.logger.info("Cooldown ended. Resuming object detection.")
                    result = None
                    display_frame = self.draw_hud(frame, result)

                # -------------------------------------------------------------
                # State 2 & 3: IDLE or DEBOUNCING
                # -------------------------------------------------------------
                elif self.state in (SortingStateMachine.IDLE, SortingStateMachine.DEBOUNCING):
                    # Run vision detection within ROI
                    result = self.detector.detect(frame)
                    base_display = result.annotated_frame if result else frame

                    if result:
                        # Reset empty counter
                        self.empty_count = 0

                        if result.class_name == self.candidate_class:
                            self.consecutive_count += 1
                        else:
                            # New candidate object detected
                            self.candidate_class = result.class_name
                            self.candidate_bin = result.bin_id
                            self.consecutive_count = 1
                            self.state = SortingStateMachine.DEBOUNCING

                        # Check if debounce requirement met
                        if self.consecutive_count >= CONFIG.debounce.required_consecutive_detections:
                            # Trigger Sorting Sequence!
                            self.state = SortingStateMachine.SORTING
                            self.logger.info(
                                f">> Object Confirmed: {self.candidate_class.upper()} "
                                f"(Target: Bin {self.candidate_bin}). Triggering Arm Cycle!"
                            )

                            # Render HUD before blocking arm execution
                            hud = self.draw_hud(base_display, result)
                            if not self.headless:
                                cv2.imshow(window_name, hud)
                                cv2.waitKey(1)

                            # Execute blocking 10-step pick-and-place sequence
                            t_start = time.time()
                            success = self.arm.move_to_bin(self.candidate_bin)
                            duration = time.time() - t_start

                            # Update metrics
                            status_str = "SUCCESS" if success else "FAILED"
                            if success:
                                self.sort_tally[self.candidate_bin] += 1
                                self.last_sorted_info = f"{self.candidate_class.upper()} -> Bin {self.candidate_bin}"

                            # Log to console and CSV
                            self.logger.log_sort_event(
                                object_class=self.candidate_class,
                                bin_id=self.candidate_bin,
                                duration_sec=duration,
                                status=status_str,
                                confidence=result.confidence,
                            )

                            # Enter cooldown period
                            self.state = SortingStateMachine.COOLDOWN
                            self.cooldown_end_time = time.time() + CONFIG.debounce.post_sort_cooldown_sec
                            self.candidate_class = None
                            self.consecutive_count = 0
                            continue

                    else:
                        # No object detected inside pickup zone
                        self.empty_count += 1
                        if self.empty_count >= CONFIG.debounce.empty_frames_to_reset:
                            self.state = SortingStateMachine.IDLE
                            self.candidate_class = None
                            self.consecutive_count = 0

                    display_frame = self.draw_hud(base_display, result)

                else:
                    display_frame = self.draw_hud(frame, None)

                # Display GUI window
                if not self.headless:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self.logger.info("User requested exit ('q' pressed).")
                        break

        except Exception as e:
            self.logger.exception(f"Unexpected error in main loop: {e}")

        finally:
            self.cleanup(window_name)

    def cleanup(self, window_name: str):
        """Safely park arm and close video stream."""
        self.logger.info("Cleaning up resources...")
        try:
            self.arm.home()
            self.arm.disable_all_servos()
            self.arm.deinit()
        except Exception as e:
            self.logger.warning(f"Error during arm parking: {e}")

        if self.cap:
            self.cap.release()

        if not self.headless:
            cv2.destroyAllWindows()

        self.logger.info("Autonomous Object-Sorting System shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Object-Sorting Arm Orchestrator")
    parser.add_argument("--simulate", action="store_true", help="Force software simulation mode for arm controller")
    parser.add_argument("--csi", action="store_true", help="Use Jetson CSI camera pipeline (nvarguscamerasrc)")
    parser.add_argument("--camera-idx", type=int, default=0, help="USB camera device index (default: 0)")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode cycling through sample images")
    parser.add_argument(
        "--detector",
        type=str,
        choices=["auto", "yolo", "hsv"],
        default="auto",
        help="Vision detector engine: 'yolo', 'hsv', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit path to YOLO model file (.pt or .engine)",
    )
    args = parser.parse_args()

    cam_source = "csi" if args.csi else "usb"

    system = AutonomousSortingSystem(
        simulate_arm=args.simulate,
        camera_source=cam_source,
        camera_index=args.camera_idx,
        headless=args.headless,
        demo_mode=args.demo,
        detector_type=args.detector,
        model_path=args.model,
    )
    system.run()


if __name__ == "__main__":
    main()

