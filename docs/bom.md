# Bill of Materials (BOM) & Hardware Specifications

Autonomous Object-Sorting Robotic Arm running on NVIDIA Jetson Orin Nano 8GB.

---

## 1. Core Hardware Components

| Item | Component | Specification / Model | Est. Cost (USD) | Purpose / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Embedded AI Compute Board** | NVIDIA Jetson Orin Nano Developer Kit (8GB RAM, 40 TOPS) | ~$499 | Runs OpenCV vision, debounce logic, and drives I2C PWM driver directly. |
| **2** | **I2C PWM Driver** | PCA9685 16-Channel 12-Bit PWM Board | ~$5 – $8 | Offloads servo PWM pulse generation from Jetson to eliminate software timing jitter. |
| **3** | **4-DOF Robotic Arm Chassis** | Acrylic or Aluminum 4-Axis Arm Kit (Base, Shoulder, Elbow, Claw) | ~$25 – $50 | Standard hobby 4-DOF arm frame (e.g. Adeept, SainSmart, or generic 4-DOF laser-cut kit). |
| **4** | **Camera Module** | **Option A (CSI):** Raspberry Pi Camera v2 (Sony IMX219) with 22-pin ribbon cable<br>**Option B (USB):** Standard 720p/1080p UVC USB Webcam | ~$15 – $30 | Overlooks the fixed pickup tray. USB camera requires zero setup; CSI uses hardware ISP. |
| **5** | **Servo Actuators** | **Option A (Micro):** 4x SG90 9g Servos<br>**Option B (High-Torque):** 4x MG996R Metal Gear Servos | ~$10 (SG90)<br>~$25 (MG996R) | Joint motors for Base, Shoulder, Elbow, and Gripper. |
| **6** | **Jetson Main Power** | Official 19V / 2.37A (45W) DC Barrel Jack Power Adapter | Included with DevKit | Powers the Jetson Orin Nano. |
| **7** | **Servo Auxiliary Power Supply** | Dedicated 5V DC Regulated Power Supply (5V/2.5A+ for SG90; 5V/5A for MG996R) | ~$10 – $15 | Connected directly to PCA9685 screw terminal. Prevents Jetson brownouts. |
| **8** | **Decoupling Capacitor** | 1000 µF 16V or 25V Aluminum Electrolytic Capacitor | ~$0.50 | Soldered across PCA9685 V+ and GND terminals to smooth inrush voltage drops. |
| **9** | **Interconnect Wiring** | 4x Female-to-Female Dupont Jumper Wires (20cm) | ~$2 | Connects Jetson 40-pin header (Pins 1, 3, 5, 9) to PCA9685 logic pins. |
| **10** | **Sorting Workpiece & Tray** | 3x Colored Sorting Cubes (Red, Green, Blue ~25–30mm) + White Pickup Tray | ~$5 | Standard colored wood blocks, acrylic cubes, or 3D-printed sorting cubes. |
| **11** | **Destination Bins** | 3x Small Plastic Cups or Cardboard Bins (Labeled 1, 2, 3) | ~$3 | Placed at 45°, 90°, and 135° radially around the arm base. |

**Total Estimated Hardware Cost (excluding Jetson developer kit):** ~$65 – $140

---

## 2. Servo Selection Guide: SG90 vs MG996R

| Feature | SG90 Micro (9g) | MG996R Metal Gear (55g) |
| :--- | :--- | :--- |
| **Gear Material** | Nylon / Plastic | Brass / Metal Alloy |
| **Stall Torque** | ~1.8 kg·cm (at 4.8V) | ~10.0 kg·cm (at 4.8V) – 11.0 kg·cm (at 6.0V) |
| **Operating Current** | ~150–250 mA typical | ~500–900 mA typical |
| **Stall Current** | ~800 mA | ~2.5 A each! |
| **Pulse Width Range** | 500 µs – 2400 µs | 500 µs – 2500 µs |
| **Suitability** | Best for small lightweight acrylic arms, sorting foam or light wooden blocks (< 30g). | Recommended for aluminum arm chassis, heavier payloads (up to 150g), and industrial demos. |
| **Power Supply Need** | 5V / 2A minimum | 5V / 4A to 5V / 5A minimum |
| **Config Setting in `config.py`** | `ACTIVE_SERVO_MODEL = "SG90"` | `ACTIVE_SERVO_MODEL = "MG996R"` |

---

## 3. Power Budget & Current Consumption Breakdown

### Typical 4-Joint Operation (Simultaneous Movement)
- **4x SG90:**
  - Running current: $4 \times 0.20\text{ A} = 0.8\text{ A}$
  - Peak inrush current during acceleration: Up to $2.0\text{ A}$
  - **Required Supply:** 5V / 2.5A or 5V / 3A DC power brick.
- **4x MG996R:**
  - Running current: $4 \times 0.70\text{ A} = 2.8\text{ A}$
  - Peak inrush current during acceleration: Up to $4.5\text{ A}$
  - **Required Supply:** 5V / 5A DC power supply (or 10A buck converter with 12V input).
