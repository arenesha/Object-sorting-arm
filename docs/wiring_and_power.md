# Electrical Wiring & Power Distribution Guide

**System:** Autonomous Object-Sorting Robotic Arm  
**Controller:** NVIDIA Jetson Orin Nano Developer Kit (8GB)  
**PWM Driver:** PCA9685 16-Channel 12-bit I2C PWM Driver  
**Actuators:** 4-DOF Robotic Arm (Base, Shoulder, Elbow, Gripper)

---

## 1. Safety & Power Architecture Overview

> [!CAUTION]
> **NEVER POWER SERVOS DIRECTLY FROM THE JETSON 5V PINS!**  
> Servos—even micro SG90 models—draw significant peak current (inrush and stall currents up to 800 mA each for SG90; up to 2.5 A each for MG996R). When multiple joints accelerate simultaneously, the sudden current spike causes the Jetson's internal 5V rail to drop below the brownout threshold, immediately resetting the system or causing flash filesystem corruption.

The system uses a **dual-isolated power architecture**:
1. **Jetson Orin Nano:** Powered by its official DC barrel jack (9V–20V / 19V 45W supply).
2. **PCA9685 Logic:** Powered by the Jetson's **3.3V rail (Pin 1)**.
3. **Servo Power Rail (V+):** Powered exclusively by a dedicated **external 5V supply** (5V/2A+ for SG90, 5V/4A–5V/5A for MG996R) connected to the PCA9685 high-current screw terminal.
4. **Common Ground:** Jetson Ground and External Power Ground **must be tied together** at the PCA9685 board.

---

## 2. Jetson 40-Pin Header to PCA9685 Wiring

The Jetson Orin Nano features a standard 40-pin expansion header (similar to Raspberry Pi pinout). We utilize **I2C Bus 1**:

| Jetson 40-Pin Header Pin | Signal Name | PCA9685 Logic Pin | Wire Color Recommendation | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Pin 1** | 3.3V DC Power | **VCC** | Red | Powers the PCA9685 logic chip |
| **Pin 3** | I2C1_SDA (Data) | **SDA** | Green / Yellow | I2C Data bus line |
| **Pin 5** | I2C1_SCL (Clock) | **SCL** | Blue | I2C Clock bus line |
| **Pin 9** | GND (Ground) | **GND** | Black | Logic & reference ground |
| *(None)* | — | **OE** | — | Output Enable (leave disconnected / active low) |

*(Note: On the PCA9685 logic header, leave the `V+` pin disconnected; servo power is fed via the screw terminal below).*

---

## 3. PCA9685 Screw Terminal (External Servo Power)

The screw terminal block on the end of the PCA9685 board routes power directly to the servo headers:

| Terminal Mark | Connects To | Specification |
| :--- | :--- | :--- |
| **V+ (Positive)** | External Power Supply **+5V** | Regulated 5.0V DC (minimum 2.5A for SG90; 4.0A–5.0A for MG996R) |
| **GND (Negative)** | External Power Supply **GND** | Shared common ground with Jetson Pin 9 |

### Decoupling Capacitor Recommendation
Most PCA9685 boards provide a dedicated footprint (labeled `C1`) for a large bulk capacitor. Solder a **1000 µF / 16V (or 25V) electrolytic capacitor** across the `V+` and `GND` power terminals (observe correct polarity: negative stripe to GND). This absorbs transient voltage sags caused by sudden servo motor movement.

---

## 4. Servo Channel Connections (PCA9685 Header 0–15)

Each 3-pin servo cable connects to a 3-pin column on the PCA9685 board:
- **Yellow / White / Light Orange wire:** PWM Signal (Top row)
- **Red wire:** Power V+ 5V (Middle row)
- **Brown / Black wire:** Ground GND (Bottom row)

| Channel | Joint Name | Physical Function | Angle Range |
| :--- | :--- | :--- | :--- |
| **Channel 0** | **Base** | Waist rotation (left / center / right) | 0° to 180° |
| **Channel 1** | **Shoulder** | Lower arm elevation / pitch | 15° to 165° |
| **Channel 2** | **Elbow** | Forearm reach / pitch | 15° to 165° |
| **Channel 3** | **Gripper** | End-effector jaw clamp | 20° to 100° |

*(Channels 4 through 15 are left available for future additions, such as a camera tilt servo or indicator LEDs).*

---

## 5. Wiring Schematic Diagram

```
+-------------------------------------------------------------------------+
|                       NVIDIA JETSON ORIN NANO                           |
|                                                                         |
|   Pin 1 (3.3V) -----> Red wire ----------------------+                  |
|   Pin 3 (SDA)  -----> Green wire ----------------+   |                  |
|   Pin 5 (SCL)  -----> Blue wire -------------+   |   |                  |
|   Pin 9 (GND)  -----> Black wire --------+   |   |   |                  |
+------------------------------------------|---|---|---|------------------+
                                           |   |   |   |
                                           v   v   v   v
                                         +---------------+
                                         | GND SCL SDA VCC
                                         |               |
                                         |    PCA9685    |
   +----------------------+              |   16-CH I2C   |
   | External 5V Supply   |              |  PWM DRIVER   |
   |                      |              |               |
   |   5V (+) -----------> [ V+  ]       |  [0] [1] [2] [3]
   |                      |  Terminal    |   |   |   |   |
   |   GND (-) ----------> [ GND ]       +---|---|---|---+
   +----------------------+ (Common GND)     |   |   |   |
                                             |   |   |   +--> Gripper Servo (Ch 3)
                                             |   |   +------> Elbow Servo (Ch 2)
                                             |   +----------> Shoulder Servo (Ch 1)
                                             +--------------> Base Servo (Ch 0)
```

---

## 6. Software Verification & Bus Testing on Jetson

Once wired, verify that the Jetson can communicate with the PCA9685 over I2C:

1. **List I2C Buses:**
   ```bash
   sudo i2cdetect -l
   ```
2. **Scan Bus 1 for the PCA9685 (Address 0x40):**
   ```bash
   sudo i2cdetect -y -r 1
   ```
   **Expected Output:**
   ```
        0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
   00:          -- -- -- -- -- -- -- -- -- -- -- -- -- 
   10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
   20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
   30: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
   40: 40 -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
   50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
   60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
   70: 70 -- -- -- -- -- -- --
   ```
   *(Note: `0x40` is the board's I2C address, and `0x70` is the PCA9685 all-call address. If you see `40`, your wiring is 100% verified!)*

3. **Grant User I2C Permissions (Avoid running everything with `sudo`):**
   ```bash
   sudo usermod -aG i2c $USER
   ```
   Log out and log back in to apply group permissions.
