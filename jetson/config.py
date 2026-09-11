"""Autonomous Object-Sorting Robotic Arm - Central Configuration

NVIDIA Jetson Orin Nano (8GB) + PCA9685 I2C 16-Channel PWM Driver.
All timing, joint angles, I2C mappings, HSV vision thresholds,
and camera pipeline settings are centralized here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ==============================================================================
# 1. CAMERA CONFIGURATION
# ==============================================================================
@dataclass
class CameraConfig:
    """Camera capture settings supporting USB webcams and Jetson CSI cameras."""
    # Source type: 'usb' or 'csi'
    source_type: str = "usb"
    
    # USB Camera index (e.g., 0 for /dev/video0)
    device_index: int = 0
    
    # Native CSI sensor capture resolution (IMX219 / IMX477 native mode)
    capture_width: int = 1280
    capture_height: int = 720
    
    # Processed frame resolution & frame rate for vision pipeline
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30
    
    # GStreamer pipeline helper for Raspberry Pi Camera v2 / IMX219 / IMX477 CSI on Jetson
    sensor_id: int = 0
    flip_method: int = 0  # 0: none, 2: 180 deg rotate

    def get_gstreamer_pipeline(self) -> str:
        """Returns the optimized GStreamer pipeline string for Jetson hardware nvarguscamerasrc.
        
        Captures at a native CSI sensor mode (1280x720) in hardware NVMM memory,
        scales down via Jetson's hardware nvvidconv to frame_width x frame_height (640x480),
        and feeds into OpenCV appsink with max-buffers=1 to eliminate frame latency.
        """
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){self.capture_width}, height=(int){self.capture_height}, "
            f"format=(string)NV12, framerate=(fraction){self.fps}/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, width=(int){self.frame_width}, height=(int){self.frame_height}, format=(string)BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        )


def get_camera_backend(source_type: str = "usb") -> int:
    """Returns the optimal OpenCV capture backend based on platform and camera type."""
    import sys
    import cv2
    if source_type.lower() == "csi":
        return cv2.CAP_GSTREAMER
    if sys.platform.startswith("win"):
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


# ==============================================================================
# 2. I2C & PCA9685 SERVO DRIVER CONFIGURATION
# ==============================================================================
@dataclass
class I2CConfig:
    """I2C Bus & PCA9685 PWM Driver hardware parameters."""
    # Jetson Orin Nano 40-pin header:
    # Pin 3 = I2C1_SDA, Pin 5 = I2C1_SCL -> Bus 1 (or Bus 7 on some carrier boards)
    i2c_bus: int = 1
    
    # Default PCA9685 address (all address jumpers A0-A5 open = 0x40)
    pca9685_address: int = 0x40
    
    # Standard analog servo frequency (50Hz = 20ms period)
    pwm_frequency_hz: int = 50


@dataclass
class ServoHardwarePreset:
    """Pulse-width configuration tailored to physical servo model."""
    name: str
    min_pulse_us: int
    max_pulse_us: int
    actuation_range_deg: float = 180.0


# Preset profiles for popular robotic arm servos
SERVO_PRESETS = {
    "SG90": ServoHardwarePreset(name="SG90 Micro", min_pulse_us=500, max_pulse_us=2400, actuation_range_deg=180.0),
    "MG996R": ServoHardwarePreset(name="MG996R High Torque", min_pulse_us=500, max_pulse_us=2500, actuation_range_deg=180.0),
    "STANDARD": ServoHardwarePreset(name="Generic 1-2ms", min_pulse_us=1000, max_pulse_us=2000, actuation_range_deg=180.0),
}

# Active servo model (change to "MG996R" if using metal gear servos)
ACTIVE_SERVO_MODEL = "SG90"


@dataclass
class ServoChannelMap:
    """PCA9685 channel mapping (0-15) for each joint of the 4-DOF arm."""
    base: int = 0       # Joint 1: Waist / Turntable rotation
    shoulder: int = 1   # Joint 2: Lower arm / Shoulder pitch
    elbow: int = 2      # Joint 3: Forearm / Elbow pitch
    gripper: int = 3    # Joint 4: End-effector clamp / Claw


# Software safe angle limits [min_deg, max_deg] to prevent physical over-travel / collisions
SERVO_LIMITS = {
    "base": (0.0, 180.0),
    "shoulder": (15.0, 165.0),
    "elbow": (15.0, 165.0),
    "gripper": (20.0, 100.0),  # Clamped to prevent servo stalling at extreme close/open
}


# ==============================================================================
# 3. ARM POSE & MOTION CONFIGURATION
# ==============================================================================
@dataclass
class ArmPose:
    """Defines the joint angles (in degrees) for a target robot posture."""
    base: float
    shoulder: float
    elbow: float
    gripper: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "base": self.base,
            "shoulder": self.shoulder,
            "elbow": self.elbow,
            "gripper": self.gripper,
        }


# Gripper clamp angles
GRIPPER_OPEN = 30.0    # Wide enough to clear sample sorting block
GRIPPER_CLOSED = 85.0  # Firm grip on block without stalling motor

# Key Arm Poses:
# 1. HOME: Safe neutral posture (tucked up, compact footprint)
POSE_HOME = ArmPose(
    base=90.0,
    shoulder=135.0,
    elbow=45.0,
    gripper=GRIPPER_OPEN,
)

# 2. PICKUP: Lowered posture directly reaching down to the fixed tray pickup zone
POSE_PICKUP = ArmPose(
    base=90.0,
    shoulder=60.0,
    elbow=120.0,
    gripper=GRIPPER_OPEN,
)

# 3. TRANSIT: Arm lifted upwards to clear bin rims and obstacles while rotating
POSE_TRANSIT = ArmPose(
    base=90.0,
    shoulder=115.0,
    elbow=70.0,
    gripper=GRIPPER_CLOSED,
)

# 4. BIN PRESET DESTINATIONS (Base angles for sorting bins)
# Bin 1 (Red)   -> Base = 45 degrees (Left)
# Bin 2 (Green) -> Base = 90 degrees (Straight / Rear or Extended)
# Bin 3 (Blue)  -> Base = 135 degrees (Right)
BIN_CONFIG: Dict[int, Dict[str, any]] = {
    1: {
        "name": "Bin 1 (Red)",
        "color_class": "red",
        "base_angle": 45.0,
        "shoulder_drop": 75.0,
        "elbow_drop": 105.0,
    },
    2: {
        "name": "Bin 2 (Green)",
        "color_class": "green",
        "base_angle": 90.0,
        "shoulder_drop": 80.0,
        "elbow_drop": 100.0,
    },
    3: {
        "name": "Bin 3 (Blue)",
        "color_class": "blue",
        "base_angle": 135.0,
        "shoulder_drop": 75.0,
        "elbow_drop": 105.0,
    },
}

# Motion interpolation & timing parameters
@dataclass
class MotionConfig:
    """Speed and smoothing parameters for servo trajectory generation."""
    # Step increment per interpolation interval in degrees
    step_deg: float = 1.5
    # Delay between trajectory interpolation steps (seconds)
    step_delay_sec: float = 0.015
    # Settle time after reaching key waypoint (seconds)
    settle_time_sec: float = 0.35
    # Gripper actuation delay to ensure physical grip/release completes (seconds)
    gripper_delay_sec: float = 0.45


# ==============================================================================
# 4. VISION & DETECTION CONFIGURATION
# ==============================================================================
@dataclass
class VisionConfig:
    """HSV color filtering, ROI coordinates, and contour thresholds."""
    # Region of Interest (ROI) for the fixed pickup tray: (x, y, width, height)
    # Calibrated for 640x480 resolution (centered pickup zone on table)
    roi_pickup_zone: Tuple[int, int, int, int] = (220, 260, 200, 180)

    # Minimum and maximum contour areas (in pixels) within ROI to accept an object
    min_contour_area: int = 1200
    max_contour_area: int = 25000

    # Morphological noise filter kernel size (odd integer)
    morph_kernel_size: int = 5
    gaussian_blur_ksize: int = 5

    # HSV Color Boundaries (Hue: 0-179, Sat: 0-255, Val: 0-255)
    # Red wraps around 0/180 in HSV space -> uses two sub-ranges
    hsv_ranges: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = field(
        default_factory=lambda: {
            "red": [
                ((0, 120, 70), (10, 255, 255)),     # Lower red hue band
                ((170, 120, 70), (180, 255, 255)),  # Upper red hue band
            ],
            "green": [
                ((36, 80, 70), (86, 255, 255)),
            ],
            "blue": [
                ((95, 110, 70), (135, 255, 255)),
            ],
        }
    )

    # Color class to Bin ID mapping
    color_to_bin: Dict[str, int] = field(
        default_factory=lambda: {
            "red": 1,
            "green": 2,
            "blue": 3,
        }
    )


# ==============================================================================
# 5. DEBOUNCE & SYSTEM ORCHESTRATION CONFIGURATION
# ==============================================================================
@dataclass
class DebounceConfig:
    """State-machine debouncing to eliminate false triggers and transient noise."""
    # Number of consecutive identical classification frames required before arm triggers
    required_consecutive_detections: int = 5
    
    # Cooldown duration (seconds) after completing a pick-and-place before accepting new objects
    # Gives the user time to place the next object without accidental double-triggering
    post_sort_cooldown_sec: float = 2.5
    
    # Number of frames with no object before resetting the debounce counter
    empty_frames_to_reset: int = 3


# ==============================================================================
# 6. DYNAMIC YOLO MODEL CONFIGURATION
# ==============================================================================
import os
import json

# Dedicated folder where user uploads/drops .pt or .engine weights
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
MODELS_CONFIG_FILE = os.path.join(MODELS_DIR, "models_config.json")


def get_available_models() -> List[str]:
    """Scan models directory and return list of valid .pt and .engine model files."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
        return []
    valid_exts = (".pt", ".engine", ".onnx")
    return [
        f for f in os.listdir(MODELS_DIR)
        if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(MODELS_DIR, f))
    ]


def get_active_model_info() -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve the active model filename and its full filesystem path.
    Looks up models_config.json; if not configured, automatically selects
    the most recently modified model in MODELS_DIR.
    
    Returns:
        (model_filename, full_path) or (None, None) if no models found.
    """
    available = get_available_models()
    if not available:
        return None, None

    # Check if models_config.json specifies an active model
    selected_name = None
    if os.path.exists(MODELS_CONFIG_FILE):
        try:
            with open(MODELS_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                configured = data.get("active_model")
                if configured and configured in available:
                    selected_name = configured
        except Exception:
            pass

    # Fallback: pick the most recently modified model file
    if not selected_name:
        available.sort(
            key=lambda f: os.path.getmtime(os.path.join(MODELS_DIR, f)),
            reverse=True,
        )
        selected_name = available[0]

    return selected_name, os.path.join(MODELS_DIR, selected_name)


@dataclass
class YOLOConfig:
    """Settings for dynamic YOLO object detection."""
    models_dir: str = MODELS_DIR
    models_config_file: str = MODELS_CONFIG_FILE
    confidence_threshold: float = 0.50
    # Restrict detection box centroid strictly inside pickup zone ROI
    restrict_to_pickup_zone: bool = True
    # Device: '0' for Jetson GPU/CUDA, or 'cpu'
    device: str = "0"


# ==============================================================================
# 7. GLOBAL CONFIGURATION BUNDLE INSTANCE
# ==============================================================================
class AppConfig:
    """Master application configuration container."""
    camera: CameraConfig = CameraConfig()
    i2c: I2CConfig = I2CConfig()
    channels: ServoChannelMap = ServoChannelMap()
    active_servo_preset: ServoHardwarePreset = SERVO_PRESETS[ACTIVE_SERVO_MODEL]
    motion: MotionConfig = MotionConfig()
    vision: VisionConfig = VisionConfig()
    yolo: YOLOConfig = YOLOConfig()
    debounce: DebounceConfig = DebounceConfig()


# Singleton config instance
CONFIG = AppConfig()

