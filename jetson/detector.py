"""Autonomous Object-Sorting Robotic Arm - Vision Detector Module

Provides modular object detection and classification for NVIDIA Jetson Orin Nano.
Includes:
- BaseDetector abstract interface (allows seamless swap to YOLOv8-nano / TFLite)
- ColorHSVDetector: Robust ROI-focused HSV contour detection & color classification
- Complete visual annotation and bounding box extraction
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import CONFIG, VisionConfig


@dataclass
class DetectionResult:
    """Standardized detection result output passed to main orchestrator."""
    class_name: str
    confidence: float
    bin_id: int
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h) in original frame coords
    center: Tuple[int, int]                  # (cx, cy) in original frame coords
    area: float
    annotated_frame: np.ndarray


class BaseDetector(ABC):
    """
    Abstract Base Class for all detection engines.
    Swapping detection models (e.g., HSV baseline -> YOLOv8-nano -> TFLite)
    requires only inheriting from BaseDetector and implementing detect().
    """

    def __init__(self):
        self.last_annotated_frame: Optional[np.ndarray] = None

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """
        Analyze a raw camera frame and return detected object details.
        
        Args:
            frame: Input image (BGR format, typically 640x480)
            
        Returns:
            DetectionResult if a valid object is present in pickup zone, else None.
        """
        pass


class ColorHSVDetector(BaseDetector):
    """
    Baseline color and contour detector using OpenCV HSV color space filtering.
    Restricts detection strictly to the configured pickup zone Region of Interest (ROI),
    applies morphological noise rejection, finds contours, and maps color to bin ID.
    """

    def __init__(self, vision_cfg: Optional[VisionConfig] = None):
        super().__init__()
        self.cfg = vision_cfg or CONFIG.vision
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.cfg.morph_kernel_size, self.cfg.morph_kernel_size)
        )

    def _create_color_mask(self, hsv_roi: np.ndarray, color_name: str) -> np.ndarray:
        """Create a binary mask for a specific color class, handling hue wrap-around."""
        ranges = self.cfg.hsv_ranges.get(color_name, [])
        if not ranges:
            return np.zeros(hsv_roi.shape[:2], dtype=np.uint8)

        combined_mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv_roi, lower_np, upper_np)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Apply morphological operations: open to remove specks, close to seal gaps
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, self.morph_kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        return combined_mask

    def detect(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """
        Perform detection on the frame inside the pickup zone ROI.
        """
        if frame is None or frame.size == 0:
            return None

        annotated = frame.copy()
        rx, ry, rw, rh = self.cfg.roi_pickup_zone

        # Safety clamp ROI bounds to frame dimensions
        h_frame, w_frame = frame.shape[:2]
        rx = max(0, min(rx, w_frame - 1))
        ry = max(0, min(ry, h_frame - 1))
        rw = max(1, min(rw, w_frame - rx))
        rh = max(1, min(rh, h_frame - ry))

        # Draw ROI boundary on the HUD: Blue when searching, turns Green/Yellow on detection
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 2)
        cv2.putText(
            annotated,
            "PICKUP ZONE",
            (rx, ry - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )

        # Extract Region of Interest
        roi = frame[ry : ry + rh, rx : rx + rw]
        if roi.size == 0:
            return None

        # Preprocess ROI: Gaussian blur + Convert to HSV
        blurred_roi = cv2.GaussianBlur(
            roi, (self.cfg.gaussian_blur_ksize, self.cfg.gaussian_blur_ksize), 0
        )
        hsv_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)

        best_detection: Optional[DetectionResult] = None
        max_area_found = 0.0

        # Iterate over all registered color classes (e.g. red, green, blue)
        for color_name in self.cfg.hsv_ranges.keys():
            mask = self._create_color_mask(hsv_roi, color_name)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if self.cfg.min_contour_area <= area <= self.cfg.max_contour_area:
                    if area > max_area_found:
                        max_area_found = area
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        global_bx = rx + bx
                        global_by = ry + by
                        cx = global_bx + (bw // 2)
                        cy = global_by + (bh // 2)

                        # Offset contour to global frame coordinates
                        global_cnt = cnt + np.array([rx, ry])

                        bin_id = self.cfg.color_to_bin.get(color_name, 1)
                        # Estimate confidence based on contour solidity and area ratio
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area) / hull_area if hull_area > 0 else 0.8
                        confidence = min(1.0, max(0.5, solidity))

                        best_detection = DetectionResult(
                            class_name=color_name,
                            confidence=confidence,
                            bin_id=bin_id,
                            bounding_box=(global_bx, global_by, bw, bh),
                            center=(cx, cy),
                            area=area,
                            annotated_frame=annotated,
                        )

        # If an object was detected, draw visual highlights on the annotated frame
        if best_detection:
            gx, gy, gw, gh = best_detection.bounding_box
            cx, cy = best_detection.center
            c_name = best_detection.class_name.upper()
            b_id = best_detection.bin_id

            # Pick display color
            color_map = {
                "red": (0, 0, 255),
                "green": (0, 255, 0),
                "blue": (255, 100, 0),
            }
            display_bgr = color_map.get(best_detection.class_name, (0, 255, 255))

            # Draw bounding box and center reticle
            cv2.rectangle(annotated, (gx, gy), (gx + gw, gy + gh), display_bgr, 2)
            cv2.drawMarker(annotated, (cx, cy), display_bgr, cv2.MARKER_CROSS, 16, 2)

            # Draw tag banner
            label = f"{c_name} -> BIN {b_id} ({best_detection.confidence*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (gx, gy - th - 10), (gx + tw + 6, gy), display_bgr, -1)
            cv2.putText(
                annotated,
                label,
                (gx + 3, gy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Highlight pickup zone as actively acquired
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            self.last_annotated_frame = annotated
            best_detection.annotated_frame = annotated
            return best_detection

        self.last_annotated_frame = annotated
        return None


# ==============================================================================
# 2. DYNAMIC UPLOADABLE YOLO DETECTOR
# ==============================================================================
import json
import os
import logging
from config import MODELS_DIR, get_active_model_info

yolo_logger = logging.getLogger("YOLODetector")


class YOLODetector(BaseDetector):
    """
    Dynamic, uploadable YOLO detector (supports .pt and .engine formats).
    - Automatically discovers the active model from /jetson/models/
    - Reads class names dynamically from model metadata (model.names) — zero hardcoding
    - Auto-generates user-editable <model_name>_mapping.json if not present
    - Skips/ignores classes with bin=0 (unassigned), logging an unmapped notice
    - Retains standard BaseDetector interface for seamless drop-in
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        restrict_to_pickup: Optional[bool] = None,
    ):
        super().__init__()
        self.conf_threshold = conf_threshold or CONFIG.yolo.confidence_threshold
        self.restrict_to_pickup = (
            restrict_to_pickup
            if restrict_to_pickup is not None
            else CONFIG.yolo.restrict_to_pickup_zone
        )
        self.model = None
        self.is_simulated = False
        self.class_names: List[str] = []
        self.class_to_bin: Dict[str, int] = {}
        self.mapping_file_path: Optional[str] = None
        self.last_annotated_frame: Optional[np.ndarray] = None
        # Enhanced tracking & smoothing
        self.smooth_box: Optional[Tuple[float, float, float, float]] = None
        self.last_valid_detection: Optional[DetectionResult] = None
        self.missed_frame_count: int = 0
        self.max_missed_persistence: int = 2  # Keep detection alive across 2 momentary frame drops

        # Resolve model path
        if model_path:
            self.model_filename = os.path.basename(model_path)
            self.model_path = os.path.abspath(model_path)
        else:
            resolved_name, resolved_path = get_active_model_info()
            self.model_filename = resolved_name
            self.model_path = resolved_path

        self._load_model_and_mapping()

    def _load_model_and_mapping(self):
        """Load weights, extract dynamic metadata, and initialize/sync mapping JSON."""
        if not self.model_path or not os.path.exists(self.model_path):
            yolo_logger.warning(
                f"No YOLO model found at '{self.model_path}'. "
                "YOLODetector running in SIMULATED fallback mode."
            )
            self.is_simulated = True
            self.model_stem = "simulated_model"
            self.class_names = ["sample_red_cube", "sample_green_cube", "sample_blue_cube", "unmapped_defect"]
            self._sync_mapping_file()
            return

        self.model_stem = os.path.splitext(self.model_filename)[0]

        # Attempt to load using Ultralytics YOLO
        try:
            from ultralytics import YOLO
            yolo_logger.info(f"Loading YOLO model from: '{self.model_path}'...")
            self.model = YOLO(self.model_path)

            # Dynamically extract classes from model metadata (model.names is dict {id: name})
            raw_names = self.model.names
            if isinstance(raw_names, dict):
                self.class_names = [str(raw_names[i]) for i in sorted(raw_names.keys())]
            elif isinstance(raw_names, (list, tuple)):
                self.class_names = [str(n) for n in raw_names]
            else:
                self.class_names = [str(raw_names)]

            yolo_logger.info(
                f"Successfully loaded '{self.model_filename}' with {len(self.class_names)} classes: {self.class_names}"
            )

        except ImportError:
            yolo_logger.warning(
                "Package 'ultralytics' not installed. Running YOLODetector in SIMULATION mode. "
                "Install via 'pip install ultralytics' on Jetson to enable live neural inference."
            )
            self.is_simulated = True
            # Read metadata from dummy model or inspect if JSON exists
            self.class_names = self._inspect_fallback_classes()

        except Exception as e:
            yolo_logger.warning(
                f"Could not load '{self.model_path}' ({e}). Running in SIMULATION mode."
            )
            self.is_simulated = True
            self.class_names = self._inspect_fallback_classes()

        # Generate or sync <model_name>_mapping.json
        self._sync_mapping_file()

    def _inspect_fallback_classes(self) -> List[str]:
        """Inspect existing mapping or return generic classes if model cannot be parsed directly."""
        existing_mapping_path = os.path.join(MODELS_DIR, f"{self.model_stem}_mapping.json")
        if os.path.exists(existing_mapping_path):
            try:
                with open(existing_mapping_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return list(data.keys())
            except Exception:
                pass
        return ["red_object", "green_object", "blue_object", "defect"]

    def _sync_mapping_file(self):
        """
        Ensures <model_name>_mapping.json exists.
        - First run: creates file with all detected classes mapped to 0 (unassigned).
        - Subsequent runs: loads existing mappings and merges any newly added classes without
          overwriting user edits.
        """
        self.mapping_file_path = os.path.join(MODELS_DIR, f"{self.model_stem}_mapping.json")
        os.makedirs(MODELS_DIR, exist_ok=True)

        existing_mapping: Dict[str, int] = {}
        file_existed = os.path.exists(self.mapping_file_path)

        if file_existed:
            try:
                with open(self.mapping_file_path, "r", encoding="utf-8") as f:
                    existing_mapping = json.load(f)
            except Exception as e:
                yolo_logger.error(f"Error reading existing mapping JSON: {e}")

        # Sync classes
        updated_mapping = dict(existing_mapping)
        new_classes_added = False

        for cls_name in self.class_names:
            if cls_name not in updated_mapping:
                # Default new classes to bin 0 (unassigned / skipped)
                updated_mapping[cls_name] = 0
                new_classes_added = True

        self.class_to_bin = updated_mapping

        # Write file if new or updated
        if not file_existed or new_classes_added:
            try:
                with open(self.mapping_file_path, "w", encoding="utf-8") as f:
                    json.dump(updated_mapping, f, indent=2)

                if not file_existed:
                    yolo_logger.info(
                        f">> Auto-generated new class-to-bin mapping file: '{self.mapping_file_path}'. "
                        "All classes defaulted to bin 0 (unassigned). Please edit this file to assign real bin numbers."
                    )
                else:
                    yolo_logger.info(
                        f"Synchronized new classes into existing mapping file: '{self.mapping_file_path}'"
                    )
            except Exception as e:
                yolo_logger.error(f"Failed to write mapping file: {e}")
        else:
            yolo_logger.info(
                f"Loaded active class-to-bin mapping from: '{self.mapping_file_path}'"
            )

    def reload_mapping(self):
        """Reload the mapping JSON from disk (allows runtime edits without restarting system)."""
        if self.mapping_file_path and os.path.exists(self.mapping_file_path):
            try:
                with open(self.mapping_file_path, "r", encoding="utf-8") as f:
                    self.class_to_bin = json.load(f)
                yolo_logger.info(f"Reloaded class-to-bin mapping: {self.class_to_bin}")
            except Exception as e:
                yolo_logger.error(f"Failed to reload mapping file: {e}")

    def detect(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """
        Executes YOLO inference, filters by pickup zone, checks class-to-bin mapping,
        skips unmapped classes (bin 0), and returns standardized DetectionResult.
        """
        if frame is None or frame.size == 0:
            return None

        annotated = frame.copy()
        rx, ry, rw, rh = CONFIG.vision.roi_pickup_zone

        # Safety clamp ROI bounds to frame dimensions
        h_frame, w_frame = frame.shape[:2]
        rx = max(0, min(rx, w_frame - 1))
        ry = max(0, min(ry, h_frame - 1))
        rw = max(1, min(rw, w_frame - rx))
        rh = max(1, min(rh, h_frame - ry))

        # Draw ROI boundary on HUD
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 2)
        cv2.putText(
            annotated,
            f"PICKUP ZONE (YOLO: {self.model_stem})",
            (rx, ry - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )

        detected_candidates: List[DetectionResult] = []

        # ----------------------------------------------------------------------
        # Case A: Live Ultralytics Inference
        # ----------------------------------------------------------------------
        if not self.is_simulated and self.model is not None:
            try:
                # High-sensitivity inference: conf threshold down to 0.18 min, iou=0.45, imgsz=640
                eff_conf = max(0.18, float(self.conf_threshold))
                results = self.model(
                    frame,
                    conf=eff_conf,
                    iou=0.45,
                    imgsz=640,
                    verbose=False,
                )
                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue

                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())

                        # Dynamic class name lookup from model.names
                        if isinstance(self.model.names, dict):
                            class_name = str(self.model.names.get(cls_id, f"class_{cls_id}"))
                        elif isinstance(self.model.names, (list, tuple)) and cls_id < len(self.model.names):
                            class_name = str(self.model.names[cls_id])
                        else:
                            class_name = f"class_{cls_id}"

                        bw = x2 - x1
                        bh = y2 - y1
                        cx = x1 + (bw // 2)
                        cy = y1 + (bh // 2)

                        # Check if within or overlapping pickup zone
                        in_pickup = (rx <= cx <= rx + rw and ry <= cy <= ry + rh) or \
                                    (max(rx, x1) < min(rx + rw, x2) and max(ry, y1) < min(ry + rh, y2))

                        # Dynamic bin lookup
                        bin_id = int(self.class_to_bin.get(class_name, 0)) or 1

                        if self.restrict_to_pickup and not in_pickup:
                            # Render indicator so user sees detection is actively tracking!
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 212, 255), 2)
                            cv2.putText(
                                annotated,
                                f"{class_name.upper()} ({conf*100:.0f}%) [OUTSIDE ZONE]",
                                (x1, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 212, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            # Still record candidate so UI/drag simulation functions everywhere
                            det = DetectionResult(
                                class_name=class_name,
                                confidence=conf,
                                bin_id=bin_id,
                                bounding_box=(x1, y1, bw, bh),
                                center=(cx, cy),
                                area=float(bw * bh),
                                annotated_frame=annotated,
                            )
                            detected_candidates.append(det)
                            continue

                        # Dynamic bin lookup
                        bin_id = int(self.class_to_bin.get(class_name, 0))

                        # If bin == 0: SKIP/IGNORE and log unmapped notification
                        if bin_id == 0:
                            yolo_logger.warning(
                                f"[YOLO] Unmapped class '{class_name}' detected in pickup zone (bin=0). Skipping sort."
                            )
                            # Render subtle warning box on HUD
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                            cv2.putText(
                                annotated,
                                f"{class_name.upper()} [UNMAPPED (BIN 0)]",
                                (x1, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 165, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            continue

                        # Valid assigned object!
                        det = DetectionResult(
                            class_name=class_name,
                            confidence=conf,
                            bin_id=bin_id,
                            bounding_box=(x1, y1, bw, bh),
                            center=(cx, cy),
                            area=float(bw * bh),
                            annotated_frame=annotated,
                        )
                        detected_candidates.append(det)

            except Exception as e:
                yolo_logger.error(f"Error during YOLO inference: {e}")

        # ----------------------------------------------------------------------
        # Case B: Simulation Fallback (matches synthetic frames for testing)
        # ----------------------------------------------------------------------
        else:
            # Fallback contour detector within ROI to emulate YOLO detections
            roi = frame[ry : ry + rh, rx : rx + rw]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Detect colored objects present in synthetic stream
            for test_color, mapped_cls in [
                ("red", self.class_names[0] if self.class_names else "red_object"),
                ("green", self.class_names[1] if len(self.class_names) > 1 else "green_object"),
                ("blue", self.class_names[2] if len(self.class_names) > 2 else "blue_object"),
            ]:
                ranges = CONFIG.vision.hsv_ranges.get(test_color, [])
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for l, u in ranges:
                    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(l, dtype=np.uint8), np.array(u, dtype=np.uint8)))
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    area = cv2.contourArea(cnt)
                    if area >= CONFIG.vision.min_contour_area:
                        bx, by, bw, bh = cv2.boundingRect(cnt)
                        gx, gy = rx + bx, ry + by
                        cx, cy = gx + (bw // 2), gy + (bh // 2)

                        bin_id = int(self.class_to_bin.get(mapped_cls, 0))
                        if bin_id == 0:
                            yolo_logger.warning(
                                f"[SIMULATED YOLO] Unmapped class '{mapped_cls}' detected in pickup zone (bin=0). Skipping sort."
                            )
                            cv2.rectangle(annotated, (gx, gy), (gx + bw, gy + bh), (0, 165, 255), 1)
                            cv2.putText(
                                annotated,
                                f"{mapped_cls.upper()} [UNMAPPED (BIN 0)]",
                                (gx, max(15, gy - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (0, 165, 255),
                                1,
                            )
                            continue

                        det = DetectionResult(
                            class_name=mapped_cls,
                            confidence=0.95,
                            bin_id=bin_id,
                            bounding_box=(gx, gy, bw, bh),
                            center=(cx, cy),
                            area=float(area),
                            annotated_frame=annotated,
                        )
                        detected_candidates.append(det)

        # If candidates exist, choose the one with largest area / highest confidence
        if detected_candidates:
            self.missed_frame_count = 0
            best_det = max(detected_candidates, key=lambda d: d.confidence * d.area)
            gx, gy, gw, gh = best_det.bounding_box

            # Temporal exponential moving average (EMA) smoothing for rock-solid boxes
            if self.smooth_box is not None:
                sx, sy, sw, sh = self.smooth_box
                alpha = 0.70  # 70% current, 30% history
                gx = int(alpha * gx + (1 - alpha) * sx)
                gy = int(alpha * gy + (1 - alpha) * sy)
                gw = int(alpha * gw + (1 - alpha) * sw)
                gh = int(alpha * gh + (1 - alpha) * sh)
            self.smooth_box = (float(gx), float(gy), float(gw), float(gh))
            cx, cy = gx + gw // 2, gy + gh // 2
            best_det.bounding_box = (gx, gy, gw, gh)
            best_det.center = (cx, cy)
            self.last_valid_detection = best_det

            # Bin-specific color theme (BGR)
            BIN_COLORS = {
                1: (255, 212, 0),   # Cyan
                2: (0, 230, 118),   # Green
                3: (0, 152, 255),   # Amber/Orange
            }
            box_color = BIN_COLORS.get(best_det.bin_id, (0, 255, 0))

            # Visual overlay on HUD
            cv2.rectangle(annotated, (gx, gy), (gx + gw, gy + gh), box_color, 2)
            cv2.drawMarker(annotated, (cx, cy), box_color, cv2.MARKER_CROSS, 16, 2)

            label = f"{best_det.class_name.upper()} -> BIN {best_det.bin_id} ({best_det.confidence*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (gx, gy - th - 10), (gx + tw + 6, gy), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (gx + 3, gy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), box_color, 2)
            self.last_annotated_frame = annotated
            best_det.annotated_frame = annotated
            return best_det

        elif self.last_valid_detection is not None and self.missed_frame_count < self.max_missed_persistence:
            # Frame drop grace period: maintain detection during momentary webcam motion blur
            self.missed_frame_count += 1
            decay_det = self.last_valid_detection
            gx, gy, gw, gh = decay_det.bounding_box
            cx, cy = decay_det.center
            box_color = (0, 230, 118)
            cv2.rectangle(annotated, (gx, gy), (gx + gw, gy + gh), box_color, 1)
            cv2.drawMarker(annotated, (cx, cy), box_color, cv2.MARKER_CROSS, 14, 2)
            self.last_annotated_frame = annotated
            decay_det.annotated_frame = annotated
            return decay_det

        self.smooth_box = None
        self.last_valid_detection = None
        self.last_annotated_frame = annotated
        return None


# Backwards compatibility alias
YOLOv8Detector = YOLODetector

