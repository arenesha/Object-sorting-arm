# Servo & Vision Calibration Guide

This guide walks through the physical assembly, electrical zeroing, angle calibration, and vision tuning procedures for the 4-DOF Autonomous Object-Sorting Robotic Arm.

---

## 1. Pre-Assembly Servo Zeroing (Crucial Step!)

> [!IMPORTANT]
> **DO NOT attach servo horns to the arm chassis before zeroing the motors!**  
> If you attach servo horns at random angles and later power on the system, the servos may instantly whip to 90°, forcing the arm beyond its physical mechanical stops and stripping internal gears.

### Zeroing Procedure:
1. Connect the PCA9685 to the Jetson I2C bus and connect external 5V power to the PCA9685 screw terminal (see [wiring_and_power.md](file:///c:/Users/Admin/Documents/object_sorting/docs/wiring_and_power.md)).
2. Plug the 4 bare servos into Channels 0, 1, 2, and 3 on the PCA9685.
3. Run the automated centering command from your terminal:
   ```bash
   python test/test_arm_standalone.py --center
   ```
4. All 4 servos are now precisely commanded to **90.0°** (the geometric midpoint of their 180° travel range).
5. Now mount the horns to the arm frame with the joints aligned to their neutral mid-range:
   - **Base (Ch 0):** Horn pointing straight ahead (parallel with the front pickup tray).
   - **Shoulder (Ch 1):** Lower arm link pointing vertically upright (~90° to base).
   - **Elbow (Ch 2):** Forearm link pointing horizontally forward (~90° to lower arm).
   - **Gripper (Ch 3):** Jaw half-open (~45°–60°).

---

## 2. Joint Angle Tuning in `jetson/config.py`

Every laser-cut acrylic or aluminum arm chassis has slight geometric variations. You can customize the named angle constants in `jetson/config.py` to match your physical setup.

Use the interactive CLI tool:
```bash
python test/test_arm_standalone.py
```
Select **Option [6] (Test Individual Joint Angle)** to adjust each joint incrementally, noting down the exact angle values that achieve optimal positioning.

### Key Poses to Tune:

### A. HOME Posture (`POSE_HOME`)
- **Purpose:** Compact rest posture when waiting for objects.
- **Typical values:**
  ```python
  POSE_HOME = ArmPose(
      base=90.0,      # Facing forward
      shoulder=135.0,  # Folded back
      elbow=45.0,      # Tucked downward
      gripper=30.0     # Open
  )
  ```

### B. PICKUP Posture (`POSE_PICKUP`)
- **Purpose:** Reaches down directly into the fixed tray zone.
- **Tuning Tip:** Lower the elbow and shoulder gradually until the gripper jaws rest ~5–10 mm above the tray surface.
  ```python
  POSE_PICKUP = ArmPose(
      base=90.0,
      shoulder=60.0,
      elbow=120.0,
      gripper=30.0
  )
  ```

### C. TRANSIT Height (`POSE_TRANSIT`)
- **Purpose:** Lifts the object high enough to clear bin rims and obstacles while rotating the base.
  ```python
  POSE_TRANSIT = ArmPose(
      base=90.0,
      shoulder=115.0,
      elbow=70.0,
      gripper=85.0     # Closed
  )
  ```

### D. Bin Drop Angles (`BIN_CONFIG`)
- **Purpose:** Position of each sorting container around the arm base.
  ```python
  BIN_CONFIG = {
      1: {"name": "Bin 1 (Red)",   "base_angle": 45.0,  "shoulder_drop": 75.0, "elbow_drop": 105.0},
      2: {"name": "Bin 2 (Green)", "base_angle": 90.0,  "shoulder_drop": 80.0, "elbow_drop": 100.0},
      3: {"name": "Bin 3 (Blue)",  "base_angle": 135.0, "shoulder_drop": 75.0, "elbow_drop": 105.0},
  }
  ```

---

## 3. Gripper Clamp Calibration

> [!WARNING]
> Setting `GRIPPER_CLOSED` too tightly will cause the servo motor to stall against the sorting object. Stalled servos draw continuous peak stall current (up to 800 mA for SG90 / 2.5 A for MG996R), rapidly overheating the motor winding and buzzing loudly.

1. Test the grip on your physical sorting object (e.g. 25mm cube):
   - In `test_arm_standalone.py`, select **Option [6]** for `gripper`.
   - Start at angle `70.0°` and step up in `2°` increments.
   - Stop as soon as the jaw holds the cube firmly without motor buzzing or frame flexing.
2. Update `GRIPPER_OPEN` and `GRIPPER_CLOSED` in `jetson/config.py`:
   ```python
   GRIPPER_OPEN = 30.0    # Jaws open sufficiently wide
   GRIPPER_CLOSED = 82.0  # Firm grip without stalling
   ```

---

## 4. Vision & Pickup Zone Calibration

### A. Calibrating Pickup Zone ROI (`roi_pickup_zone`)
1. Place the camera overhead overlooking the tray.
2. In `jetson/config.py`, locate `roi_pickup_zone = (x, y, width, height)`.
3. The cyan rectangle on the camera HUD indicates the pickup zone. Adjust coordinates until the rectangle tightly frames your marked pickup spot.

### B. Tuning HSV Color Bounds for Your Room Lighting
Ambient lighting varies widely between fluorescent, LED, and sunlight. To dial in exact color thresholds:

1. Capture an image of your sorting object on the tray:
   ```bash
   # Or use one of the test images in test/sample_images/
   python test/test_detection_offline.py --tune test/sample_images/red_cube_center.jpg
   ```
2. Adjust the OpenCV GUI sliders:
   - **H Min / H Max:** Isolates the color hue (Red: 0–10 or 170–180; Green: 35–85; Blue: 95–135).
   - **S Min:** Filters out washed-out whites/greys (typically 70–100).
   - **V Min:** Filters out dark shadows (typically 60–80).
   - **Min Area:** Filters out small dust specks (e.g., 1000–1500 pixels).
3. Copy the output values into `jetson/config.py` under `VisionConfig.hsv_ranges`.

---

## 5. Motion Speed & Settle Time Tuning

In `jetson/config.py` under `MotionConfig`:
- `step_deg` (default `1.5°`): Smaller values yield smoother movement; larger values move faster.
- `step_delay_sec` (default `0.015s`): Time between trajectory interpolation steps.
- `settle_time_sec` (default `0.35s`): Pause after reaching waypoints before next action.
- `gripper_delay_sec` (default `0.45s`): Time for physical jaws to complete grip before lifting.
