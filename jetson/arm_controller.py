"""Autonomous Object-Sorting Robotic Arm - Arm Controller

Drives a 4-DOF robotic arm directly from NVIDIA Jetson Orin Nano via PCA9685 I2C.
Implements:
- PCA9685 hardware initialization with automatic simulation fallback
- Synchronous multi-joint smooth interpolated motion (no jerking or current spikes)
- Full 10-step Pick -> Transit -> Place -> Return Home sequence
- Safe angle clamping and emergency home/parking routines
"""

import time
import math
import logging
from typing import Dict, Optional, Tuple

from config import (
    CONFIG,
    ArmPose,
    POSE_HOME,
    POSE_PICKUP,
    POSE_TRANSIT,
    BIN_CONFIG,
    GRIPPER_OPEN,
    GRIPPER_CLOSED,
    SERVO_LIMITS,
)

try:
    from sound_effects import play_pick_sound, play_place_sound
except ImportError:
    try:
        from jetson.sound_effects import play_pick_sound, play_place_sound
    except ImportError:
        def play_pick_sound(): pass
        def play_place_sound(): pass

logger = logging.getLogger("ArmController")


class ArmController:
    """Controls the 4-DOF robotic arm via PCA9685 I2C or high-fidelity simulation."""

    def __init__(self, simulate: bool = False):
        """
        Initialize the Arm Controller.
        
        Args:
            simulate: If True, bypasses I2C hardware and runs in software simulation mode.
        """
        self.simulate = simulate
        self.pca = None
        self.servos: Dict[str, any] = {}
        
        # Track current physical angle state for all 4 joints
        # Initialize at POSE_HOME
        self.current_angles: Dict[str, float] = {
            "base": POSE_HOME.base,
            "shoulder": POSE_HOME.shoulder,
            "elbow": POSE_HOME.elbow,
            "gripper": POSE_HOME.gripper,
        }

        self._init_hardware()

    def _init_hardware(self):
        """Initialize the I2C bus and PCA9685 PWM driver with graceful fallback."""
        if self.simulate:
            logger.info("[SIMULATION MODE] Arm controller running in software simulation.")
            return

        try:
            # Import Adafruit Blinka / CircuitPython libraries
            import busio
            import board
            from adafruit_pca9685 import PCA9685
            from adafruit_motor import servo

            logger.info(
                f"Initializing PCA9685 on I2C bus {CONFIG.i2c.i2c_bus} (target address: 0x{CONFIG.i2c.pca9685_address:02X})..."
            )
            # Resolve SCL and SDA pins for Jetson 40-pin header (Pins 3 & 5)
            scl_pin = getattr(board, f"SCL_{CONFIG.i2c.i2c_bus}", None) or getattr(board, "SCL", None)
            sda_pin = getattr(board, f"SDA_{CONFIG.i2c.i2c_bus}", None) or getattr(board, "SDA", None)
            if scl_pin is None or sda_pin is None:
                scl_pin = getattr(board, "SCL", None)
                sda_pin = getattr(board, "SDA", None)

            if scl_pin is None or sda_pin is None:
                raise RuntimeError("Could not resolve SCL/SDA pins in Adafruit Blinka for this platform.")

            i2c = busio.I2C(scl_pin, sda_pin)
            self.pca = PCA9685(i2c, address=CONFIG.i2c.pca9685_address)
            self.pca.frequency = CONFIG.i2c.pwm_frequency_hz

            preset = CONFIG.active_servo_preset
            logger.info(f"Using servo preset: '{preset.name}' ({preset.min_pulse_us}us - {preset.max_pulse_us}us)")

            # Initialize joint servos on configured PCA9685 channels
            ch = CONFIG.channels
            channel_map = {
                "base": ch.base,
                "shoulder": ch.shoulder,
                "elbow": ch.elbow,
                "gripper": ch.gripper,
            }

            for joint_name, channel_num in channel_map.items():
                self.servos[joint_name] = servo.Servo(
                    self.pca.channels[channel_num],
                    min_pulse=int(preset.min_pulse_us),
                    max_pulse=int(preset.max_pulse_us),
                    actuation_range=int(preset.actuation_range_deg),
                )
            logger.info("PCA9685 hardware initialized successfully on all 4 channels.")

        except PermissionError as perm_err:
            logger.error(
                f"[I2C PERMISSION DENIED] Cannot access I2C bus ({perm_err}).\n"
                "  >> FIX: Add your user to the 'i2c' group on Linux/Jetson:\n"
                "     sudo usermod -aG i2c $USER\n"
                "  >> Then log out and log back in, or run with appropriate permissions."
            )
            logger.warning("Switching to simulation mode due to permissions error.")
            self.simulate = True

        except (ImportError, NotImplementedError) as dep_err:
            logger.info(
                f"CircuitPython / PCA9685 libraries not installed or unsupported on this OS ({dep_err}). "
                "Running in software SIMULATION mode."
            )
            self.simulate = True

        except OSError as os_err:
            logger.warning(
                f"I2C Hardware Error ({os_err}). Possible causes:\n"
                "  1. PCA9685 not powered (VCC 3.3V / GND from Jetson 40-pin header).\n"
                "  2. SDA/SCL pins swapped (Jetson Pin 3=SDA, Pin 5=SCL).\n"
                f"  3. Wrong I2C bus index (current config: bus {CONFIG.i2c.i2c_bus}). Check 'i2cdetect -y -r 1'.\n"
                f"  4. PCA9685 address is not 0x{CONFIG.i2c.pca9685_address:02X}.\n"
                "Switching automatically to high-fidelity SIMULATION mode."
            )
            self.simulate = True

        except Exception as err:
            logger.warning(
                f"Physical PCA9685 hardware not accessible ({err}). "
                "Switching automatically to high-fidelity SIMULATION mode."
            )
            self.simulate = True

    def _clamp_angle(self, joint_name: str, angle: float) -> float:
        """Clamp angle to safe configured physical limits to prevent mechanical strain."""
        min_deg, max_deg = SERVO_LIMITS.get(joint_name, (0.0, 180.0))
        clamped = max(min_deg, min(max_deg, float(angle)))
        if clamped != angle:
            logger.warning(
                f"Angle {angle:.1f}° for joint '{joint_name}' clamped to safe limit: {clamped:.1f}°"
            )
        return clamped

    def set_joint_angle_instant(self, joint_name: str, angle: float):
        """
        Directly command a single servo to an angle (used internally during interpolation).
        """
        clamped_angle = self._clamp_angle(joint_name, angle)
        self.current_angles[joint_name] = clamped_angle

        if not self.simulate and joint_name in self.servos:
            try:
                self.servos[joint_name].angle = clamped_angle
            except Exception as e:
                logger.error(f"Error writing to servo '{joint_name}': {e}")
        else:
            # Simulation logging for debugging
            pass

    def move_joints_interpolated(
        self,
        target_angles: Dict[str, float],
        step_delay: Optional[float] = None,
        step_deg: Optional[float] = None,
    ):
        """
        Smoothly interpolate all joints simultaneously from current angles to target angles.
        
        Calculates the maximum displacement among joints and steps all joints concurrently
        using proportional linear interpolation. This prevents current surges, physical jerking,
        and kinematic collisions.
        """
        step_delay = step_delay or CONFIG.motion.step_delay_sec
        step_deg = step_deg or CONFIG.motion.step_deg

        # Validate and clamp all target angles
        sanitized_targets: Dict[str, float] = {}
        for joint, target in target_angles.items():
            if joint in self.current_angles:
                sanitized_targets[joint] = self._clamp_angle(joint, target)

        # Determine the maximum angular distance needed across all joints
        start_angles = {j: self.current_angles[j] for j in sanitized_targets}
        deltas = {j: sanitized_targets[j] - start_angles[j] for j in sanitized_targets}
        max_delta = max((abs(d) for d in deltas.values()), default=0.0)

        if max_delta < 0.5:
            # Target is already effectively reached
            for j, tgt in sanitized_targets.items():
                self.set_joint_angle_instant(j, tgt)
            return

        # Calculate number of discrete interpolation steps
        num_steps = max(1, int(math.ceil(max_delta / step_deg)))

        for step in range(1, num_steps + 1):
            fraction = step / num_steps
            # Linear stepping fraction (can be smoothed with cosine ease if desired)
            for joint, delta in deltas.items():
                interpolated_angle = start_angles[joint] + (delta * fraction)
                self.set_joint_angle_instant(joint, interpolated_angle)
            time.sleep(step_delay)

        # Ensure exact target is set at completion
        for joint, target in sanitized_targets.items():
            self.set_joint_angle_instant(joint, target)

    def move_to_pose(self, pose: ArmPose, step_delay: Optional[float] = None):
        """Move arm smoothly to a predefined named ArmPose."""
        self.move_joints_interpolated(pose.as_dict(), step_delay=step_delay)
        time.sleep(CONFIG.motion.settle_time_sec)

    def open_gripper(self):
        """Actuate gripper to OPEN angle."""
        self.move_joints_interpolated(
            {"gripper": GRIPPER_OPEN},
            step_deg=CONFIG.motion.step_deg * 2.0,
        )
        time.sleep(CONFIG.motion.gripper_delay_sec)

    def close_gripper(self):
        """Actuate gripper to CLOSED angle."""
        self.move_joints_interpolated(
            {"gripper": GRIPPER_CLOSED},
            step_deg=CONFIG.motion.step_deg * 2.0,
        )
        time.sleep(CONFIG.motion.gripper_delay_sec)

    def home(self):
        """
        Return the arm smoothly to the tucked HOME posture.
        Performs a staged motion: lifts shoulder/elbow first, then centers base.
        """
        logger.info("Moving arm to HOME position...")
        # Step A: Lift arm upwards safely while maintaining base
        self.move_joints_interpolated({
            "shoulder": POSE_HOME.shoulder,
            "elbow": POSE_HOME.elbow,
        })
        # Step B: Align base to 90 degrees and reset gripper
        self.move_joints_interpolated({
            "base": POSE_HOME.base,
            "gripper": POSE_HOME.gripper,
        })
        time.sleep(CONFIG.motion.settle_time_sec)
        logger.info("Arm safely parked at HOME position.")

    def move_to_bin(self, bin_id: int) -> bool:
        """
        Executes the exact 10-step Autonomous Pick-and-Place sequence:
        
        1. Start at home position
        2. Open gripper
        3. Lower arm to the FIXED pickup position
        4. Close gripper (grip object)
        5. Lift arm to a safe transit height
        6. Rotate base to the angle preset for the target bin ID
        7. Lower arm over the bin
        8. Open gripper (release object)
        9. Lift arm, rotate base back, return to home position
        10. Ready to resume detection (blocking call completes)
        
        Args:
            bin_id: Target bin destination identifier (e.g., 1, 2, or 3)
            
        Returns:
            True if sequence completed successfully, False otherwise.
        """
        if bin_id not in BIN_CONFIG:
            logger.error(f"Invalid bin_id {bin_id}. Available bins: {list(BIN_CONFIG.keys())}")
            return False

        bin_info = BIN_CONFIG[bin_id]
        target_base_angle = bin_info["base_angle"]
        shoulder_drop = bin_info["shoulder_drop"]
        elbow_drop = bin_info["elbow_drop"]

        logger.info(f"=== Beginning Sort Cycle: Transferring item to {bin_info['name']} (Base: {target_base_angle}°) ===")
        start_time = time.time()

        try:
            # Step 1: Verify starting at or transition to home position
            logger.info("Step 1/10: Verifying Home position")
            self.home()

            # Step 2: Open gripper before descending
            logger.info("Step 2/10: Opening gripper")
            self.open_gripper()

            # Step 3: Lower arm to the FIXED pickup position
            logger.info("Step 3/10: Lowering arm to fixed pickup position")
            self.move_joints_interpolated({
                "base": POSE_PICKUP.base,
                "shoulder": POSE_PICKUP.shoulder,
                "elbow": POSE_PICKUP.elbow,
            })
            time.sleep(CONFIG.motion.settle_time_sec)

            # Step 4: Close gripper (grip object)
            logger.info("Step 4/10: Closing gripper on object")
            play_pick_sound()
            self.close_gripper()

            # Step 5: Lift arm to safe transit height
            logger.info("Step 5/10: Lifting arm to safe transit height")
            self.move_joints_interpolated({
                "shoulder": POSE_TRANSIT.shoulder,
                "elbow": POSE_TRANSIT.elbow,
            })
            time.sleep(CONFIG.motion.settle_time_sec)

            # Step 6: Rotate base to target bin preset angle
            logger.info(f"Step 6/10: Rotating base to target bin angle ({target_base_angle}°)")
            self.move_joints_interpolated({
                "base": target_base_angle,
            })
            time.sleep(CONFIG.motion.settle_time_sec)

            # Step 7: Lower arm over the bin
            logger.info("Step 7/10: Lowering arm over target bin")
            self.move_joints_interpolated({
                "shoulder": shoulder_drop,
                "elbow": elbow_drop,
            })
            time.sleep(CONFIG.motion.settle_time_sec)

            # Step 8: Open gripper (release object into bin)
            logger.info("Step 8/10: Opening gripper to release object")
            play_place_sound()
            self.open_gripper()

            # Step 9: Lift arm, rotate base back, return to home position
            logger.info("Step 9/10: Lifting arm and returning to Home position")
            # 9a: Lift up above bin rim
            self.move_joints_interpolated({
                "shoulder": POSE_TRANSIT.shoulder,
                "elbow": POSE_TRANSIT.elbow,
            })
            # 9b: Return to home
            self.home()

            # Step 10: Ready to resume detection
            duration = time.time() - start_time
            logger.info(f"Step 10/10: Sort cycle complete in {duration:.2f}s! Ready to resume detection.")
            return True

        except Exception as e:
            logger.exception(f"Error during pick-and-place cycle: {e}")
            try:
                self.home()
            except Exception:
                pass
            return False

    def disable_all_servos(self):
        """De-energize servos (sets duty cycle to 0) to prevent resting motor heat/jitter."""
        if not self.simulate and self.pca:
            try:
                for joint_name, s in self.servos.items():
                    try:
                        s.fraction = None
                    except Exception:
                        pass
                for channel_idx in [
                    CONFIG.channels.base,
                    CONFIG.channels.shoulder,
                    CONFIG.channels.elbow,
                    CONFIG.channels.gripper,
                ]:
                    self.pca.channels[channel_idx].duty_cycle = 0
                logger.info("All PCA9685 servo channels disabled (PWM duty cycle set to 0).")
            except Exception as e:
                logger.error(f"Error disabling servos: {e}")
        else:
            logger.info("[SIMULATION] All servos disabled.")

    def deinit(self):
        """Cleanly releases PCA9685 hardware resources and de-energizes servos."""
        self.disable_all_servos()
        if not self.simulate and self.pca is not None:
            try:
                self.pca.deinit()
                logger.info("PCA9685 hardware deinitialized cleanly.")
            except Exception as e:
                logger.debug(f"Note during PCA9685 deinit: {e}")

    def __del__(self):
        try:
            self.deinit()
        except Exception:
            pass
