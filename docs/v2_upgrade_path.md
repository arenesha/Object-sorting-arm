# V2 Upgrade Path: Inverse Kinematics, Vision-Guided Pickup & Deep Learning

> [!NOTE]
> **Roadmap & Architecture Specification (Documented for Future Extension)**  
> Version 1 of this project purposefully uses a **fixed pickup position** (config-defined preset angles) to establish a bulletproof, jitter-free foundation. This document details the mathematical and software blueprints for upgrading the system to **Version 2: Dynamic Camera-Guided Inverse Kinematics (IK)**.

---

## 1. Upgrade Pillar 1: Analytical Inverse Kinematics (IK)

In V1, the arm reaches a static coordinate via hardcoded joint angles $(\theta_1, \theta_2, \theta_3)$. In V2, the arm accepts dynamic Cartesian targets $(X, Y, Z)$ anywhere within its reachable workspace volume.

### Arm Link Kinematic Geometry
- Link 1 (Base height): $L_0$ (vertical pedestal from table to shoulder pivot)
- Link 2 (Upper arm / Bicep): $L_1$ (shoulder to elbow pivot)
- Link 3 (Forearm): $L_2$ (elbow to wrist/gripper pivot)
- Link 4 (Gripper / Tool tip offset): $L_3$

### Mathematical Formulation

```
                 (Elbow)
                  O [Link 2, L2]
                 / \
   [Link 1, L1] /   \
               /     \
    (Shoulder)O       O (End-Effector / Gripper)
              |       Target: (X, Y, Z)
              | [L0]
          ____|____ (Base / Turntable)
```

#### Step A: Base Yaw Angle ($\theta_{\text{base}}$)
The base simply rotates in the horizontal $(X, Y)$ plane to point the arm towards the target:
$$\theta_{\text{base}} = \text{atan2}(Y, X)$$

#### Step B: Planar Projection $(R, Z')$
Calculate the horizontal distance in the arm's plane of motion:
$$R = \sqrt{X^2 + Y^2}$$
Offset by the base height $L_0$ and tool tip approach angle $\phi$:
$$Z' = Z - L_0 - L_3 \sin(\phi)$$
$$R' = R - L_3 \cos(\phi)$$
Distance from shoulder pivot to wrist center:
$$D = \sqrt{{R'}^2 + {Z'}^2}$$

#### Step C: Elbow Angle ($\theta_{\text{elbow}}$) via Law of Cosines
Using the triangle formed by $L_1$, $L_2$, and $D$:
$$\cos(\alpha) = \frac{L_1^2 + L_2^2 - D^2}{2 L_1 L_2}$$
$$\theta_{\text{elbow}} = 180^\circ - \arccos(\cos(\alpha))$$

#### Step D: Shoulder Angle ($\theta_{\text{shoulder}}$)
$$\beta = \text{atan2}(Z', R')$$
$$\cos(\gamma) = \frac{L_1^2 + D^2 - L_2^2}{2 L_1 D}$$
$$\gamma = \arccos(\cos(\gamma))$$
$$\theta_{\text{shoulder}} = \beta + \gamma$$

---

## 2. Upgrade Pillar 2: Camera-to-Workspace Coordinate Mapping (Homography)

To transform pixel coordinates $(u, v)$ from the camera frame into physical millimeters $(X, Y)$ on the sorting workbench:

### Planar Homography Transformation
When the camera is mounted at a fixed angle overlooking the tabletop, a $3 \times 3$ perspective transformation matrix $\mathbf{H}$ maps 2D image coordinates to 2D real-world coordinates:

$$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

### Calibration Routine:
1. Place 4 fiducial markers (or ArUco tags) at known physical coordinates on the tray:
   - $P_1 = (0, 0)\text{ mm}$
   - $P_2 = (200, 0)\text{ mm}$
   - $P_3 = (200, 150)\text{ mm}$
   - $P_4 = (0, 150)\text{ mm}$
2. Detect their pixel centers $(u_i, v_i)$ in the camera feed.
3. Compute $\mathbf{H}$ using OpenCV:
   ```python
   H, _ = cv2.findHomography(image_points, physical_points)
   ```
4. When any object is detected at centroid $(u_c, v_c)$, apply:
   ```python
   pt_img = np.array([[[u_c, v_c]]], dtype=np.float32)
   pt_real = cv2.perspectiveTransform(pt_img, H)
   real_x, real_y = pt_real[0][0]
   ```
5. Feed $(real\_x, real\_y, Z_{\text{table}})$ directly to the IK solver!

---

## 3. Upgrade Pillar 3: TensorRT Deep Learning Object Detection (YOLOv8)

The current architecture is built with a modular `BaseDetector` interface in `jetson/detector.py`, designed specifically so YOLOv8-nano can be dropped in without modifying `main.py` or the state machine.

### Integration Steps:
1. **Export YOLOv8 to TensorRT Engine on Jetson:**
   ```bash
   pip install ultralytics
   yolo export model=yolov8n.pt format=engine device=0 half=True
   ```
   *(Generates `yolov8n.engine` leveraging the Jetson Orin Nano's Ampere GPU Tensor Cores for ~60+ FPS inference).*

2. **Implement `YOLOv8Detector` in `jetson/detector.py`:**
   ```python
   from ultralytics import YOLO

   class YOLOv8Detector(BaseDetector):
       def __init__(self, model_path="yolov8n.engine"):
           self.model = YOLO(model_path)
           self.class_to_bin = {
               "apple": 1,
               "bottle": 2,
               "can": 3,
           }

       def detect(self, frame: np.ndarray) -> Optional[DetectionResult]:
           results = self.model.predict(frame, conf=0.5, verbose=False)
           # Extract bounding boxes, centroid, and class name...
           # Return DetectionResult(...)
   ```

3. **Switch detector in `main.py`:**
   ```python
   # Replace:
   # self.detector = ColorHSVDetector()
   # With:
   self.detector = YOLOv8Detector("yolov8n.engine")
   ```

---

## 4. Upgrade Pillar 4: Conveyor Belt & Dynamic Motion Tracking

For moving objects (e.g., a mini motorized conveyor belt):
1. Use an optical flow tracker or Kalman filter to predict object position at time $t_{\text{intercept}} = t_{\text{current}} + \Delta t_{\text{arm\_transit}}$.
2. Command the arm to lead the object and grip dynamically on the fly.
