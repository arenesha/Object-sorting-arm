"""Autonomous Object-Sorting Robotic Arm - Standalone Arm Calibration & Test Tool

Allows direct testing and calibration of the 4-DOF arm via PCA9685 without
running vision/camera detection.
Supports both interactive CLI menu and command-line flags.

Usage:
    python test_arm_standalone.py                # Interactive CLI menu
    python test_arm_standalone.py --simulate     # Test in software simulation mode
    python test_arm_standalone.py --center       # Center all servos at 90 deg for horn installation
    python test_arm_standalone.py --cycle 1      # Run full dry-run pick-and-place cycle to Bin 1
"""

import sys
import os
import argparse
import time

# Add parent jetson directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jetson")))

from config import (
    CONFIG,
    POSE_HOME,
    POSE_PICKUP,
    POSE_TRANSIT,
    BIN_CONFIG,
    GRIPPER_OPEN,
    GRIPPER_CLOSED,
    SERVO_LIMITS,
)
from arm_controller import ArmController


def print_banner():
    print("=" * 65)
    print("      4-DOF ROBOTIC ARM - STANDALONE CALIBRATION & TEST TOOL      ")
    print("      Target: NVIDIA Jetson Orin Nano + PCA9685 I2C Driver        ")
    print("=" * 65)


def interactive_menu(arm: ArmController):
    while True:
        print("\n--- Calibration & Diagnostics Menu ---")
        print(" [1] Center all servos to 90° (Mounting / Horn Alignment)")
        print(" [2] Move to POSE_HOME (Rest posture)")
        print(" [3] Move to POSE_PICKUP (Pickup zone reach)")
        print(" [4] Move to POSE_TRANSIT (Safe transit height)")
        print(" [5] Test Gripper (Open / Close)")
        print(" [6] Test Individual Joint Angle")
        print(" [7] Test Bin Positions (Bin 1, 2, or 3)")
        print(" [8] Run Full 10-Step Pick & Place Cycle (Dry Run)")
        print(" [9] Relax / Disable All Servos (PWM Duty Cycle = 0)")
        print(" [0] Exit")

        choice = input("\nEnter choice [0-9]: ").strip()

        if choice == "1":
            print("\n>> Centering all 4 servos to 90.0°...")
            arm.move_joints_interpolated({
                "base": 90.0,
                "shoulder": 90.0,
                "elbow": 90.0,
                "gripper": 90.0,
            })
            print(">> All servos centered at 90.0°. Safe to attach mechanical horns now.")

        elif choice == "2":
            print("\n>> Moving to POSE_HOME...")
            arm.home()
            print(f">> Reached HOME: {arm.current_angles}")

        elif choice == "3":
            print("\n>> Moving to POSE_PICKUP...")
            arm.open_gripper()
            arm.move_to_pose(POSE_PICKUP)
            print(f">> Reached PICKUP: {arm.current_angles}")

        elif choice == "4":
            print("\n>> Moving to POSE_TRANSIT...")
            arm.move_to_pose(POSE_TRANSIT)
            print(f">> Reached TRANSIT: {arm.current_angles}")

        elif choice == "5":
            print("\n>> Testing Gripper Open/Close...")
            print("Opening gripper...")
            arm.open_gripper()
            time.sleep(1.0)
            print("Closing gripper...")
            arm.close_gripper()
            time.sleep(1.0)
            print("Gripper test complete.")

        elif choice == "6":
            print("\nAvailable joints: base, shoulder, elbow, gripper")
            joint = input("Enter joint name: ").strip().lower()
            if joint not in arm.current_angles:
                print(f"Error: Unknown joint '{joint}'")
                continue
            min_l, max_l = SERVO_LIMITS.get(joint, (0, 180))
            current = arm.current_angles[joint]
            val_str = input(f"Enter target angle ({min_l}° to {max_l}°, current: {current:.1f}°): ").strip()
            try:
                target_val = float(val_str)
                arm.move_joints_interpolated({joint: target_val})
                print(f">> Joint '{joint}' moved to {arm.current_angles[joint]:.1f}°")
            except ValueError:
                print("Invalid number.")

        elif choice == "7":
            print("\nAvailable Bins:")
            for b_id, b_info in BIN_CONFIG.items():
                print(f"  [{b_id}] {b_info['name']} (Base: {b_info['base_angle']}°)")
            b_sel = input("Select Bin ID [1-3]: ").strip()
            try:
                b_num = int(b_sel)
                if b_num in BIN_CONFIG:
                    b_info = BIN_CONFIG[b_num]
                    print(f">> Moving over {b_info['name']}...")
                    arm.move_to_pose(POSE_TRANSIT)
                    arm.move_joints_interpolated({"base": b_info["base_angle"]})
                    arm.move_joints_interpolated({
                        "shoulder": b_info["shoulder_drop"],
                        "elbow": b_info["elbow_drop"],
                    })
                    print(">> Over bin. Releasing object preview...")
                    arm.open_gripper()
                    time.sleep(1.0)
                    arm.home()
                else:
                    print("Invalid bin number.")
            except ValueError:
                print("Invalid input.")

        elif choice == "8":
            print("\nSelect target bin for full autonomous cycle:")
            for b_id, b_info in BIN_CONFIG.items():
                print(f"  [{b_id}] {b_info['name']}")
            b_sel = input("Enter Bin ID [1-3]: ").strip()
            try:
                b_num = int(b_sel)
                if b_num in BIN_CONFIG:
                    print(f"\n>> Starting 10-Step Cycle for Bin {b_num}...")
                    success = arm.move_to_bin(b_num)
                    print(f">> Cycle result: {'SUCCESS' if success else 'FAILED'}")
                else:
                    print("Invalid bin ID.")
            except ValueError:
                print("Invalid input.")

        elif choice == "9":
            print("\n>> De-energizing servos...")
            arm.disable_all_servos()
            print(">> Servos relaxed.")

        elif choice == "0":
            print("\n>> Returning to HOME posture and exiting...")
            try:
                arm.home()
            except Exception:
                pass
            print("Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Standalone 4-DOF Robotic Arm Calibration Tool")
    parser.add_argument("--simulate", action="store_true", help="Force software simulation mode without I2C hardware")
    parser.add_argument("--center", action="store_true", help="Command all joints to 90 degrees and exit")
    parser.add_argument("--home", action="store_true", help="Command arm to HOME position and exit")
    parser.add_argument("--cycle", type=int, choices=[1, 2, 3], help="Run a full 10-step pick-and-place cycle to specified bin")
    parser.add_argument("--test-all", action="store_true", help="Run automated dry run through all poses and bins")
    args = parser.parse_args()

    print_banner()

    arm = ArmController(simulate=args.simulate)

    if args.center:
        print("Centering all joints to 90°...")
        arm.move_joints_interpolated({"base": 90, "shoulder": 90, "elbow": 90, "gripper": 90})
        print("Done.")
        return

    if args.home:
        print("Homing arm...")
        arm.home()
        print("Done.")
        return

    if args.cycle is not None:
        print(f"Executing dry-run cycle for Bin {args.cycle}...")
        success = arm.move_to_bin(args.cycle)
        print(f"Cycle finished with status: {success}")
        return

    if args.test_all:
        print("Running comprehensive automated motion test...")
        print("1. Homing...")
        arm.home()
        print("2. Pickup pose...")
        arm.open_gripper()
        arm.move_to_pose(POSE_PICKUP)
        arm.close_gripper()
        print("3. Transit pose...")
        arm.move_to_pose(POSE_TRANSIT)
        for b_id in [1, 2, 3]:
            print(f"4.{b_id} Testing Bin {b_id} cycle...")
            arm.move_to_bin(b_id)
        print("All motion tests successfully completed!")
        return

    # Interactive menu by default
    interactive_menu(arm)


if __name__ == "__main__":
    main()
