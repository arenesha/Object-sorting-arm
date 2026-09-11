# Autonomous Object-Sorting Robotic Arm
### Embedded AI & Direct Servo Robotics on NVIDIA Jetson Orin Nano (8GB)

[![Target: Jetson Orin Nano](https://img.shields.io/badge/Target-NVIDIA%20Jetson%20Orin%20Nano%208GB-76B900?logo=nvidia)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![PWM Driver: PCA9685](https://img.shields.io/badge/PWM%20Driver-PCA9685%20I2C%2016--CH-blue)](https://www.adafruit.com/product/815)
[![Language: Python 3](https://img.shields.io/badge/Language-Python%203-3776AB?logo=python)](https://www.python.org/)
[![Vision: OpenCV](https://img.shields.io/badge/Vision-OpenCV%204-5C3EE8?logo=opencv)](https://opencv.org/)

---

## 1. Project Summary

This project delivers a complete, production-grade **Autonomous Vision-Guided Object-Sorting System** running entirely on a single **NVIDIA Jetson Orin Nano (8GB)**. 

The Jetson handles **both**:
1. **Vision & Decision-Making:** Live camera capture, region-of-interest (ROI) filtering, noise suppression, color classification, and debounce verification.
2. **Direct Motion Control:** Generating smooth multi-joint trajectory interpolation and driving a 4-DOF robotic arm directly over I2C via a PCA9685 16-channel PWM driver board.

**No secondary microcontrollers (such as Arduino, ESP32, or STM32) are required.** The entire application runs as a single, high-performance Python process with zero serial handshakes or latency bottlenecks.

```
+--------------------------------------------------------------------------------+
|                        NVIDIA JETSON ORIN NANO (8GB)                           |
|                                                                                |
|  [ Camera (USB/CSI) ]                                                          |
|          |                                                                     |
|          v                                                                     |
|  [ OpenCV ColorHSVDetector ]                                                   |
|          |                                                                     |
|          v                                                                     |
|  [ Debounce State Machine ]                                                    |
|          |                                                                     |
|          v (Target Bin ID)                                                     |
|  [ ArmController: Multi-Joint Trajectory Interpolator ]                        |
|          |                                                                     |
|          v I2C Bus (SDA: Pin 3 / SCL: Pin 5)                                   |
+----------|---------------------------------------------------------------------+
           |
           v
+------------------------+             +-----------------------------------------+
|  PCA9685 16-CH DRIVER  |             |  EXTERNAL 5V POWER SUPPLY (2A–5A)       |
|  - Logic VCC: 3.3V     |             |  (Dedicated power rail for servos,      |
|  - Address: 0x40       | <---------- |   isolated from Jetson 5V rail)         |
+------------------------+             +-----------------------------------------+
     |   |   |   |
     |   |   |   +--------> Servo 3: Gripper (End-Effector Clamp)
     |   |   +------------> Servo 2: Elbow (Forearm Pitch)
     |   +----------------> Servo 1: Shoulder (Bicep Elevation)
     +--------------------> Servo 0: Base (Waist Turntable Rotation)
```

---

## 2. Hardware Architecture & Crucial Safety Warning

> [!CAUTION]
> **DO NOT DRIVE SERVOS DIRECTLY FROM JETSON SOFTWARE PWM PINS!**  
> Linux kernel task scheduling and thread context switches induce jitter on software PWM pins, causing erratic servo vibration and stripped gears. Furthermore, powering servos from the Jetson's onboard 5V pins causes catastrophic current spikes and brownouts.
>
> **The Solution:**  
> 1. **PCA9685 PWM Driver:** Jetson sends lightweight angle commands over I2C (Bus 1, Pins 3 & 5). The PCA9685 generates hardware-precise 50Hz PWM pulses with zero CPU overhead.
> 2. **Isolated External Power:** Servos are powered via a dedicated 5V/2A+ (SG90) or 5V/5A (MG996R) DC supply connected to the PCA9685 screw terminal.
> 3. **Common Ground:** External GND and Jetson Pin 9 GND are tied together at the PCA9685 board.

For full schematics and pin tables, see [wiring_and_power.md](file:///c:/Users/Admin/Documents/object_sorting/docs/wiring_and_power.md).

---

## 3. The 10-Step Pick-and-Place Motion Sequence

Each sorting cycle executes an exact 10-step smooth interpolated trajectory:

1. **Verify Home:** Start at safe tucked resting posture (`POSE_HOME`).
2. **Open Gripper:** Open jaws to `GRIPPER_OPEN` (30°).
3. **Lower Arm:** Smooth multi-joint descent to the fixed pickup tray coordinates (`POSE_PICKUP`).
4. **Grip Object:** Close jaws firmly to `GRIPPER_CLOSED` (85°).
5. **Lift Arm:** Lift to safe transit elevation (`POSE_TRANSIT`) to clear bin rims and obstacles.
6. **Rotate Base:** Rotate waist joint to the designated target bin angle (Bin 1: 45°, Bin 2: 90°, Bin 3: 135°).
7. **Lower over Bin:** Lower shoulder and elbow over the sorting bin container.
8. **Release Object:** Open gripper to release the sorting block into the bin.
9. **Return to Home:** Retract arm upwards, rotate base back, and park safely at `POSE_HOME`.
10. **Resume Detection:** Blocking call completes; logging event triggers; system enters brief cooldown and resumes vision scanning.

*All motion uses synchronous multi-joint linear interpolation (`step_deg = 1.5°`, `step_delay = 15ms`) to prevent current spikes, sudden jerking, and dropped items.*

---

## 4. Repository Structure

```
object_sorting/
├── jetson/
│   ├── config.py              # Centralized configuration (angles, I2C, channels, HSV, debounce)
│   ├── detector.py             # Modular BaseDetector, ColorHSVDetector, and YOLOv8 extension
│   ├── arm_controller.py       # PCA9685 I2C driver, trajectory interpolation, 10-step sequence
│   ├── logger.py                # Formatted console output + persistent sort_history.csv logger
│   ├── main.py                  # Real-time closed-loop orchestrator with live HUD overlay
│   └── requirements.txt         # Project dependencies
├── test/
│   ├── generate_sample_data.py  # Realistic synthetic camera dataset generator
│   ├── test_detection_offline.py# Offline test suite + interactive OpenCV HSV trackbar tuner GUI
│   ├── test_arm_standalone.py   # Standalone CLI tool for servo zeroing and joint calibration
│   ├── sample_images/           # Synthetic test images (red, green, blue, empty, offsets)
│   └── output_detected/         # Visual detection evaluation output
└── docs/
    ├── README.md                # Master documentation
    ├── wiring_and_power.md      # Detailed 40-pin wiring, power isolation, and common ground
    ├── bom.md                   # Bill of Materials with parts specs and power calculations
    ├── calibration_guide.md     # Step-by-step mechanical zeroing and angle tuning guide
    └── v2_upgrade_path.md       # Roadmap: Analytical Inverse Kinematics (IK) & TensorRT YOLOv8
```

---

## 5. Quickstart Guide

### Step 1: Install System & Python Dependencies on Jetson
```bash
# Update Jetson system packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv i2c-tools

# Add current user to I2C group
sudo usermod -aG i2c $USER

# Install Python requirements
cd object_sorting/jetson
pip3 install -r requirements.txt
```

### Step 2: Verify I2C Wiring
Check that the PCA9685 is detected at address `0x40` on I2C bus 1:
```bash
sudo i2cdetect -y -r 1
```

### Step 3: Mechanical Zeroing & Angle Calibration
Before assembling your arm links, center all servos to 90°:
```bash
# From workspace root:
python test/test_arm_standalone.py --center
```
Use the interactive menu to calibrate your pickup and bin angles:
```bash
python test/test_arm_standalone.py
```
*(Select Option 6 to tune joints, Option 8 to dry-run a full sorting cycle).*

### Step 4: Verify Vision Offline
Run the offline vision detector on the synthetic dataset (or tune HSV sliders):
```bash
# Automated evaluation (100% test pass verification):
python test/test_detection_offline.py

# Interactive slider tuner for local lighting:
python test/test_detection_offline.py --tune test/sample_images/red_cube_center.jpg
```

### Step 5: Launch the Complete Autonomous System
```bash
# Standard run with USB webcam:
python jetson/main.py

# With Jetson CSI camera (Raspberry Pi Camera v2 / IMX219):
python jetson/main.py --csi

# Headless / background service mode (no GUI window):
python jetson/main.py --headless

# Simulation / Demo mode (runs synthetic camera feed and simulated arm):
python jetson/main.py --simulate --demo

# Run with dynamic YOLO object detection:
python jetson/main.py --detector yolo
```

---

## 6. How to Add a New Trained YOLO Model (.pt / .engine)

The system supports **100% dynamic YOLO object detection** with **zero hardcoded model paths, class names, or bin mappings**.

Follow these 4 simple steps to add and activate any custom model:

### Step 1: Drop your model into `jetson/models/`
Copy your trained PyTorch weights (`.pt`) or TensorRT engine (`.engine`) into:
```
object_sorting/jetson/models/
```
*(Example: `fruit_classifier_v1.pt` or `nuts_bolts_detector.engine`)*.

### Step 2: Activate the model using `select_model.py`
Run the model selector utility:
```bash
python jetson/select_model.py
```
- It will list all models in `jetson/models/`.
- Enter the number corresponding to your new model to activate it (or run `python jetson/select_model.py --set <filename>`).
- The script automatically reads the model's metadata (`model.names`) and creates a new mapping file:
  `jetson/models/<model_name>_mapping.json`.

### Step 3: Assign Bin Numbers in the Mapping JSON
Open the newly generated `jetson/models/<model_name>_mapping.json` in any text editor.
All detected classes start defaulted to `0` (unassigned / ignored):
```json
{
  "good_apple": 1,
  "bad_apple": 2,
  "leaf_debris": 0
}
```
- Set the target bin number (e.g., `1`, `2`, `3`) for classes you want sorted.
- Leave unwanted or distractor classes set to `0`. Objects detected as bin `0` will be **automatically skipped/ignored** by the arm and logged as unmapped notices.

### Step 4: Run the System
```bash
# Launch the orchestrator (automatically loads active YOLO model and mapping):
python jetson/main.py --detector yolo
```
You can switch models anytime simply by running `python jetson/select_model.py`!

---

## 7. Telemetry & Sort History

Every sorted object is automatically logged to both the terminal and `sort_history.csv`:
```csv
timestamp,object_class,bin_id,cycle_duration_sec,status,confidence,notes
2026-09-09T18:34:30.750088,red,1,7.39,SUCCESS,1.00,
2026-09-09T18:34:40.948134,red,1,7.42,SUCCESS,1.00,
2026-09-09T18:34:50.101057,green,2,6.37,SUCCESS,1.00,
2026-09-09T18:35:00.299655,blue,3,7.39,SUCCESS,0.99,
```

---

## 8. Extended Documentation Links

- **[Wiring & Power Distribution Guide](file:///c:/Users/Admin/Documents/object_sorting/docs/wiring_and_power.md)**
- **[Bill of Materials (BOM)](file:///c:/Users/Admin/Documents/object_sorting/docs/bom.md)**
- **[Servo & Vision Calibration Guide](file:///c:/Users/Admin/Documents/object_sorting/docs/calibration_guide.md)**
- **[V2 Upgrade Roadmap (Inverse Kinematics & YOLOv8)](file:///c:/Users/Admin/Documents/object_sorting/docs/v2_upgrade_path.md)**
