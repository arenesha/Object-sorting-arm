"""
==============================================================================
HIWONDER JETARM 2D VISUAL OVERLAY & INVERSE KINEMATICS ENGINE
==============================================================================
Provides lightweight, high-performance OpenCV rendering and 2D kinematic
trajectory generation for the Hiwonder JetArm robotic manipulator.

Designed strictly for jetson/live_demo.py OpenCV visualization.
Does NOT modify or interact with dashboard/web 2D simulation files.
==============================================================================
"""

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ==============================================================================
# UNIFIED MODERN DARK UI PALETTE & DESIGN SYSTEM (BGR FOR OPENCV)
# ==============================================================================
UI_BG_DARK = (14, 16, 20)             # Deep matte obsidian backdrop (zero brownish tint)
UI_PANEL_BG = (20, 24, 30)            # High-tech card body fill
UI_PANEL_HEADER = (28, 34, 44)        # Beveled header bar
UI_PANEL_BORDER = (42, 52, 66)        # Crisp outer panel border
UI_PANEL_BORDER_LIGHT = (64, 80, 102) # Elevated focus border
UI_RECEPTACLE_BG = (10, 12, 16)       # Deep recessed dock pocket

# High-contrast readable typography palette
UI_TEXT_TITLE = (248, 250, 254)       # Pure crisp title white
UI_TEXT_BODY = (195, 205, 218)        # Clean titanium silver
UI_TEXT_MUTED = (115, 128, 145)       # Technical muted gray

# Vivid Machine Vision Accents
UI_ACCENT_CYAN = (255, 225, 0)        # Electric Tech Cyan (BGR for #00E1FF)
UI_ACCENT_GREEN = (100, 225, 45)      # Cyber Emerald (#2DE164)
UI_ACCENT_AMBER = (0, 185, 255)       # Electric Amber (#FFA500)
UI_ACCENT_RED = (75, 75, 245)         # Neon Crimson (#F54B4B)
UI_ACCENT_BLUE = (250, 150, 30)       # Electric Cobalt (#1E96FA)

# ==============================================================================
# PHYSICAL HIWONDER JETARM HARDWARE COLOR PALETTE (MATCHING REAL ROBOT)
# ==============================================================================
# Structural links: solid mid-green anodized aluminum (#50BE46 / #52BD48)
ARM_LINK_GREEN = (72, 200, 80)        # Bright anodized lime-green body
ARM_LINK_BORDER = (32, 95, 38)        # Clean dark green bevel contour
ARM_LINK_SLOT = (42, 120, 48)         # Longitudinal bracket slot cutout
ARM_LINK_HILIGHT = (115, 230, 125)    # Specular edge highlight line

# Joint motor servos & gripper: matte black blocks (#18181A)
ARM_SERVO_BLACK = (22, 22, 24)        # Matte black block housing
ARM_SERVO_BORDER = (55, 62, 72)       # Clean border highlight for joint definition
ARM_SERVO_SHAFT = (185, 190, 198)     # Metallic silver pivot shaft / bearing

# Gripper & rubber pads
ARM_GRIPPER_FINGER = (22, 22, 24)     # Straight parallel finger plate
ARM_GRIPPER_BORDER = (65, 72, 84)     # Finger plate outline
ARM_RUBBER_PAD = (36, 38, 42)         # Textured rubber grip face


# ==============================================================================
# REUSABLE OPENCV UI DRAWING HELPER FUNCTIONS
# ==============================================================================
def draw_rounded_rect(
    canvas: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 1,
    radius: int = 8,
    fill_color: Optional[Tuple[int, int, int]] = None,
):
    """
    Draws a rectangle with smooth, anti-aliased rounded corners using OpenCV primitives.

    Args:
        canvas: Destination BGR image numpy array.
        pt1: Top-left corner (x1, y1).
        pt2: Bottom-right corner (x2, y2).
        color: Stroke border color in BGR.
        thickness: Border stroke thickness in pixels (0 for fill only).
        radius: Corner radius in pixels.
        fill_color: Optional background fill color in BGR.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    w = x2 - x1
    h = y2 - y1
    r = max(2, min(radius, w // 2, h // 2))

    # 1. Interior fill (2 overlapping rectangles + 4 corner pie slices)
    if fill_color is not None:
        cv2.rectangle(canvas, (x1 + r, y1), (x2 - r, y2), fill_color, -1)
        cv2.rectangle(canvas, (x1, y1 + r), (x2, y2 - r), fill_color, -1)
        cv2.ellipse(canvas, (x1 + r, y1 + r), (r, r), 180, 0, 90, fill_color, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x2 - r, y1 + r), (r, r), 270, 0, 90, fill_color, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x2 - r, y2 - r), (r, r), 0, 0, 90, fill_color, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (x1 + r, y2 - r), (r, r), 90, 0, 90, fill_color, -1, cv2.LINE_AA)

    # 2. Outline stroke (4 straight lines + 4 corner arc strokes)
    if thickness > 0:
        cv2.line(canvas, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_corner_brackets(
    canvas: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    bracket_len: int = 16,
    thickness: int = 2,
):
    """
    Renders high-tech camera viewfinder corner brackets at the corners of a bounding box.

    Args:
        canvas: Destination BGR image array.
        pt1: Top-left corner (x1, y1).
        pt2: Bottom-right corner (x2, y2).
        color: Stroke color in BGR.
        bracket_len: Length of the horizontal/vertical bracket arms in pixels.
        thickness: Line stroke thickness.
    """
    x1, y1 = pt1
    x2, y2 = pt2
    bl = min(bracket_len, abs(x2 - x1) // 2, abs(y2 - y1) // 2)

    # Top-Left corner
    cv2.line(canvas, (x1, y1), (x1 + bl, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y1), (x1, y1 + bl), color, thickness, cv2.LINE_AA)
    # Top-Right corner
    cv2.line(canvas, (x2, y1), (x2 - bl, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y1), (x2, y1 + bl), color, thickness, cv2.LINE_AA)
    # Bottom-Left corner
    cv2.line(canvas, (x1, y2), (x1 + bl, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y2), (x1, y2 - bl), color, thickness, cv2.LINE_AA)
    # Bottom-Right corner
    cv2.line(canvas, (x2, y2), (x2 - bl, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y2), (x2, y2 - bl), color, thickness, cv2.LINE_AA)


def draw_hud_badge(
    canvas: np.ndarray,
    text: str,
    top_left: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
    border_color: Optional[Tuple[int, int, int]] = None,
    font_scale: float = 0.38,
    font_thickness: int = 1,
    padding: Tuple[int, int] = (8, 4),
    radius: int = 5,
) -> Tuple[int, int, int, int]:
    """
    Renders a styled rounded HUD pill badge with clear text and contrast backing.

    Returns:
        (x, y, width, height) of the rendered badge plaque.
    """
    px, py = padding
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
    )
    x, y = top_left
    bw = tw + px * 2
    bh = th + py * 2 + baseline

    draw_rounded_rect(
        canvas,
        (x, y),
        (x + bw, y + bh),
        color=border_color or bg_color,
        thickness=1 if border_color else 0,
        radius=radius,
        fill_color=bg_color,
    )
    cv2.putText(
        canvas,
        text,
        (x + px, y + bh - py - baseline // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )
    return (x, y, bw, bh)


class JetArmOverlay:
    """
    Manages the Hiwonder JetArm 2D Inverse Kinematics, OpenCV rendering,
    and 10-step pick-and-place animation state machine for live_demo.py.
    """

    def __init__(
        self,
        base_pos: Tuple[int, int] = (485, 466),
        home_pos: Tuple[int, int] = (485, 225),
        pickup_pos: Tuple[int, int] = (320, 350),
        bin_positions: Optional[Dict[int, Tuple[int, int]]] = None,
        pickup_roi: Tuple[int, int, int, int] = (220, 260, 200, 180),
    ):
        self.base_pos = base_pos
        self.home_pos = home_pos
        self.pickup_pos = pickup_pos
        self.pickup_roi = pickup_roi
        self.bin_positions = bin_positions or {
            1: (110, 115),  # Bin 1 (Left)
            2: (320, 115),  # Bin 2 (Center)
            3: (530, 115),  # Bin 3 (Right)
        }

        # Current arm tip position (floats for silky-smooth interpolation)
        self.tip_x = float(home_pos[0])
        self.tip_y = float(home_pos[1])

        # Gripper state: 0.0 = fully OPEN, 1.0 = fully GRIPPED/CLOSED
        self.grip_progress = 0.0

        # Vertical lift height (0.0 = on table, 1.0 = lifted) for 3D perspective shadow
        self.lift_amount = 0.0

        # State machine states:
        # 'IDLE', 'TO_PICK', 'DESCEND', 'GRIP', 'ASCEND', 'TO_BIN',
        # 'DESCEND_BIN', 'RELEASE', 'ASCEND_BIN', 'TO_HOME', 'TRACK_DRAG'
        self.state = "IDLE"
        self.sequence: List[Dict[str, any]] = []
        self.current_step_idx = -1
        self.step_start_time = 0.0
        self.step_duration = 0.5
        self.from_pos = home_pos
        self.to_pos = home_pos
        self.target_lift = 0.0
        self.target_grip = 0.0
        self.cycle_count = 0
        self.is_running_sequence = False

        # Carried object state when clamped by gripper
        self.carried_object: Optional[Dict[str, any]] = None

        # Debounce tracking for automatic live detection trigger
        self.debounce_class: Optional[str] = None
        self.debounce_hits = 0
        self.debounce_required = 12  # frames of stable detection to auto-trigger sort
        self.auto_sort_enabled = True

        # Sort callback hook for external logging (e.g. CSV logger)
        self.on_sort_completed: Optional[Callable[[str, int, float, float], None]] = None
        self.on_bin_flash: Optional[Callable[[int], None]] = None

        # Last frame time
        self.last_time = time.time()

    # ==========================================================================
    # 1. 2D INVERSE KINEMATICS SOLVER (Planar Elbow Configuration)
    # ==========================================================================
    def solve_ik(
        self, target_x: float, target_y: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Calculates elbow (ex, ey) and wrist tip (wx, wy) coordinates from base anchor.
        Uses natural geometric bending modeled after the physical Hiwonder JetArm.
        Seamlessly transitions to/from the diagonal zigzag idle pose matching reference photo.
        """
        bx, by = float(self.base_pos[0]), float(self.base_pos[1])
        hx, hy = float(self.home_pos[0]), float(self.home_pos[1])
        dx = target_x - bx
        dy = target_y - by
        dist = math.hypot(dx, dy) or 1.0

        # Midpoint between base and target
        mx = (bx + target_x) * 0.5
        my = (by + target_y) * 0.5

        # Perpendicular normal unit vector
        px = -dy / dist
        py = dx / dist

        # Proximity to diagonal zigzag home posture
        # At home (hx, hy), home_blend = 1.0: bends inward/left (toward pickup zone side)
        # In pickup/bins (reaching forward/left), home_blend = 0.0: arches outward naturally
        home_dist = math.hypot(target_x - hx, target_y - hy)
        home_blend = max(0.0, min(1.0, 1.0 - home_dist / 140.0))

        bend_dist = min(46.0, dist * 0.28)
        bend_sign = -1.0 * home_blend + 1.0 * (1.0 - home_blend)

        elbow_x = mx + px * bend_dist * bend_sign
        elbow_y = my + py * bend_dist * bend_sign

        return (elbow_x, elbow_y), (target_x, target_y)

    # ==========================================================================
    # 2. PICK-AND-PLACE SEQUENCE GENERATOR & TRIGGERS
    # ==========================================================================
    def trigger_sort(
        self,
        pickup_pos: Tuple[int, int],
        bin_id: int,
        class_name: str = "Cube",
        conf: float = 0.95,
        carried_patch: Optional[np.ndarray] = None,
    ):
        """Build and launch the complete 9-step pick-and-place sequence."""
        if self.is_running_sequence:
            return

        bin_pos = self.bin_positions.get(bin_id, self.bin_positions[2])
        px, py = pickup_pos
        bx, by = bin_pos
        hx, hy = self.home_pos

        self.target_bin_id = bin_id
        self.target_class_name = class_name
        self.target_conf = conf

        # Color mapping by bin
        colors = {
            1: (80, 80, 220),    # Bin 1 (Soft Red)
            2: (80, 200, 100),   # Bin 2 (Soft Green)
            3: (220, 160, 60),   # Bin 3 (Soft Blue/Cyan)
        }
        obj_color = colors.get(bin_id, (0, 255, 0))

        # Store object spec to attach upon GRIP
        self.pending_object = {
            "class_name": class_name,
            "bin_id": bin_id,
            "conf": conf,
            "color": obj_color,
            "patch": carried_patch,
            "start_time": time.time(),
        }

        # Clear active carried object until gripped
        self.carried_object = None

        # Build keyframe sequence
        # Durations calibrated for natural robotic arm speeds
        self.sequence = [
            {
                "name": "TO_PICK",
                "to": (px, py - 24),
                "dur": 0.55,
                "lift": 0.0,
                "grip": 0.0,
                "status": "ARM: MOVING TO PICK",
            },
            {
                "name": "DESCEND",
                "to": (px, py),
                "dur": 0.30,
                "lift": 0.0,
                "grip": 0.0,
                "status": "ARM: DESCENDING",
            },
            {
                "name": "GRIP",
                "to": (px, py),
                "dur": 0.35,
                "lift": 0.0,
                "grip": 1.0,
                "status": "ARM: GRIPPING OBJECT",
            },
            {
                "name": "ASCEND",
                "to": (px, py - 32),
                "dur": 0.32,
                "lift": 1.0,
                "grip": 1.0,
                "status": "ARM: LIFTING",
            },
            {
                "name": "TO_BIN",
                "to": (bx, by - 22),
                "dur": 0.75,
                "lift": 1.0,
                "grip": 1.0,
                "status": f"ARM: TO BIN {bin_id}",
            },
            {
                "name": "DESCEND_BIN",
                "to": (bx, by + 12),
                "dur": 0.30,
                "lift": 0.2,
                "grip": 1.0,
                "status": f"ARM: LOWERING INTO BIN {bin_id}",
            },
            {
                "name": "RELEASE",
                "to": (bx, by + 12),
                "dur": 0.32,
                "lift": 0.0,
                "grip": 0.0,
                "status": "ARM: RELEASING OBJECT",
            },
            {
                "name": "ASCEND_BIN",
                "to": (bx, by - 26),
                "dur": 0.28,
                "lift": 0.8,
                "grip": 0.0,
                "status": "ARM: CLEARING BIN",
            },
            {
                "name": "TO_HOME",
                "to": (hx, hy),
                "dur": 0.65,
                "lift": 0.0,
                "grip": 0.0,
                "status": "ARM: RETURNING HOME",
            },
        ]

        self.current_step_idx = 0
        self.is_running_sequence = True
        self._start_step(0)

    def _start_step(self, idx: int):
        if idx >= len(self.sequence):
            # Sequence finished successfully
            self.is_running_sequence = False
            self.state = "IDLE"
            self.carried_object = None
            self.lift_amount = 0.0
            self.grip_progress = 0.0
            self.cycle_count += 1
            return

        step = self.sequence[idx]
        self.state = step["name"]
        self.from_pos = (self.tip_x, self.tip_y)
        self.to_pos = step["to"]
        self.step_duration = step["dur"]
        self.step_start_time = time.time()
        self.target_lift = step["lift"]
        self.target_grip = step["grip"]

    def _smooth_ease(self, t: float) -> float:
        """Cubic ease in-out for natural biological/mechanical servo acceleration."""
        t = max(0.0, min(1.0, t))
        return 0.5 - 0.5 * math.cos(t * math.pi)

    # ==========================================================================
    # 3. MOUSE DRAG SYNCHRONIZATION HOOKS
    # ==========================================================================
    def sync_mouse_drag_start(self, drag_center: Tuple[int, int], class_name: str, bin_id: int):
        """Latch gripper onto the object when user begins manual mouse drag."""
        self.is_running_sequence = False
        self.state = "TRACK_DRAG"
        self.grip_progress = 1.0
        self.lift_amount = 0.85
        self.tip_x = float(drag_center[0])
        self.tip_y = float(drag_center[1])
        colors = {1: (80, 80, 220), 2: (80, 200, 100), 3: (220, 160, 60)}
        self.carried_object = {
            "class_name": class_name,
            "bin_id": bin_id,
            "conf": 0.95,
            "color": colors.get(bin_id, (0, 255, 0)),
        }

    def sync_mouse_drag_update(self, drag_center: Tuple[int, int]):
        """Follow mouse position in real-time with Inverse Kinematics."""
        if self.state == "TRACK_DRAG":
            self.tip_x = float(drag_center[0])
            self.tip_y = float(drag_center[1])

    def sync_mouse_drag_drop(self, dropped_bin: Optional[int]):
        """Handle mouse release: deposit object if in bin or return home."""
        if self.state != "TRACK_DRAG":
            return

        self.state = "RELEASE"
        self.grip_progress = 0.0
        self.lift_amount = 0.0
        self.carried_object = None

        if dropped_bin is not None:
            self.cycle_count += 1
            if self.on_bin_flash:
                self.on_bin_flash(dropped_bin)

        # Smooth return to Home posture
        hx, hy = self.home_pos
        self.sequence = [
            {
                "name": "TO_HOME",
                "to": (hx, hy),
                "dur": 0.55,
                "lift": 0.0,
                "grip": 0.0,
                "status": "ARM: RETURNING HOME",
            }
        ]
        self.current_step_idx = 0
        self.is_running_sequence = True
        self._start_step(0)

    # ==========================================================================
    # 4. STATE MACHINE UPDATE PER FRAME
    # ==========================================================================
    def update(self):
        """Advances interpolation trajectories and updates joint positions."""
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        if not self.is_running_sequence:
            return

        elapsed = now - self.step_start_time
        progress = elapsed / max(0.001, self.step_duration)

        if progress >= 1.0:
            # Reached waypoint
            self.tip_x = float(self.to_pos[0])
            self.tip_y = float(self.to_pos[1])
            self.grip_progress = self.target_grip
            self.lift_amount = self.target_lift

            curr_name = self.sequence[self.current_step_idx]["name"]
            if curr_name == "GRIP":
                # Gripper closed on object: carry it!
                if hasattr(self, "pending_object") and self.pending_object:
                    self.carried_object = self.pending_object
            elif curr_name == "RELEASE":
                # Released into bin: notify completion
                if self.carried_object:
                    dur = now - self.carried_object.get("start_time", now - 1.5)
                    c_name = self.carried_object.get("class_name", "Cube")
                    b_id = self.carried_object.get("bin_id", 1)
                    conf = self.carried_object.get("conf", 0.95)
                    if self.on_sort_completed:
                        self.on_sort_completed(c_name, b_id, dur, conf)
                    if self.on_bin_flash:
                        self.on_bin_flash(b_id)
                self.carried_object = None

            # Advance to next step
            self.current_step_idx += 1
            self._start_step(self.current_step_idx)

        else:
            # Interpolate smoothly along current trajectory segment
            ease = self._smooth_ease(progress)
            fx, fy = self.from_pos
            tx, ty = self.to_pos
            self.tip_x = fx + (tx - fx) * ease
            self.tip_y = fy + (ty - fy) * ease

            curr_name = self.sequence[self.current_step_idx]["name"]
            if curr_name == "GRIP":
                self.grip_progress = ease
            elif curr_name == "RELEASE":
                self.grip_progress = 1.0 - ease

            if curr_name == "ASCEND":
                self.lift_amount = ease
            elif curr_name == "DESCEND_BIN":
                self.lift_amount = 1.0 - ease * 0.8
            elif curr_name == "ASCEND_BIN":
                self.lift_amount = 0.2 + ease * 0.6
            elif curr_name == "TO_HOME":
                self.lift_amount = max(0.0, 1.0 - ease * 1.5)

    # ==========================================================================
    # 5. HIGH-FIDELITY OPENCV HIWONDER JETARM RENDERER (POLISHED VISUALS)
    # ==========================================================================
    def draw(self, canvas: np.ndarray):
        """
        Renders the physical Hiwonder JetArm onto canvas matching actual hardware:
        - Muted, thin workspace guide lines & reach boundary arc
        - Simple flat ground drop shadow beneath wrist/carried object
        - Black base platform with chassis plate, electronics boxes & turntable bearing
        - Black rectangular shoulder servo block with green mounting brackets
        - Solid mid-green (#50BE46) upper arm structural link with bracket slot
        - Distinct black rectangular elbow servo motor housing with metallic pivot shaft
        - Solid mid-green tapered forearm link with bracket slot
        - Black wrist servo housing and RealSense camera head
        - Straight-fingered black parallel gripper with parallel translation & rubber grip pads
        - High-contrast [ OPEN ] / [ GRIP ] state indicators
        """
        bx, by = self.base_pos
        (ex, ey), (wx, wy) = self.solve_ik(self.tip_x, self.tip_y)

        # ----------------------------------------------------------------------
        # A. Dynamic Transfer Trajectory (Active during sort sequence only)
        # ----------------------------------------------------------------------
        # Keep background pristine and clutter-free during idle.
        # Only render a sleek laser guide trajectory when arm is actively transferring an item.
        if self.is_running_sequence and self.carried_object and self.target_bin_id in self.bin_positions:
            t_bin_x, t_bin_y = self.bin_positions[self.target_bin_id]
            # Draw subtle glowing trajectory beam from arm tip to target bin receptacle
            cv2.line(canvas, (int(wx), int(wy)), (int(t_bin_x), int(t_bin_y + 35)), (40, 60, 80), 1, cv2.LINE_AA)
            cv2.circle(canvas, (int(t_bin_x), int(t_bin_y + 35)), 3, UI_ACCENT_AMBER, -1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # B. Simple Flat Ground Drop Shadow
        # ----------------------------------------------------------------------
        if self.lift_amount > 0.05:
            sh_y = min(canvas.shape[0] - 6, int(wy + 12 * self.lift_amount))
            sh_x = int(wx)
            sh_rx = int(16 + 6 * self.lift_amount)
            sh_ry = int(5 + 2 * self.lift_amount)
            cv2.ellipse(canvas, (sh_x, sh_y), (sh_rx, sh_ry), 0, 0, 360, (10, 12, 16), -1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # C. Base Platform & Mounting Chassis (Stocky Black & Green Assembly)
        # ----------------------------------------------------------------------
        # 1. Black bottom mounting plate with chamfered edge
        plate_w = 84
        cv2.rectangle(
            canvas,
            (int(bx - plate_w // 2), int(by + 6)),
            (int(bx + plate_w // 2), int(by + 12)),
            ARM_SERVO_BLACK,
            -1,
        )
        cv2.rectangle(
            canvas,
            (int(bx - plate_w // 2), int(by + 6)),
            (int(bx + plate_w // 2), int(by + 12)),
            ARM_SERVO_BORDER,
            1,
        )
        # Rubber suction cup feet at ends
        for foot_x in (bx - 38, bx + 38):
            cv2.ellipse(canvas, (int(foot_x), int(by + 13)), (6, 2), 0, 0, 360, (18, 18, 20), -1, cv2.LINE_AA)

        # 2. Electronics / controller pedestal box
        # Left side: Anodized green sheet-metal bracket box
        cv2.rectangle(canvas, (int(bx - 36), int(by - 6)), (int(bx - 3), int(by + 6)), ARM_LINK_GREEN, -1)
        cv2.rectangle(canvas, (int(bx - 36), int(by - 6)), (int(bx - 3), int(by + 6)), ARM_LINK_BORDER, 1)
        # Bracket mounting holes (as seen on real JetArm)
        for hx_dot, hy_dot in ((bx - 26, by - 2), (bx - 14, by - 2), (bx - 26, by + 2), (bx - 14, by + 2)):
            cv2.circle(canvas, (int(hx_dot), int(hy_dot)), 1, (20, 24, 20), -1, cv2.LINE_AA)

        # Right side: Matte black controller / fan enclosure
        cv2.rectangle(canvas, (int(bx - 3), int(by - 6)), (int(bx + 36), int(by + 6)), ARM_SERVO_BLACK, -1)
        cv2.rectangle(canvas, (int(bx - 3), int(by - 6)), (int(bx + 36), int(by + 6)), ARM_SERVO_BORDER, 1)
        # Cooling vents
        for vx in (bx + 6, bx + 12, bx + 18):
            cv2.line(canvas, (int(vx), int(by - 3)), (int(vx), int(by + 3)), (42, 48, 56), 1, cv2.LINE_AA)
        # Power & data telemetry micro-LEDs (authentic electronic hardware look)
        cv2.circle(canvas, (int(bx + 26), int(by - 1)), 1, (0, 240, 100), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(bx + 31), int(by - 1)), 1, (255, 210, 0), -1, cv2.LINE_AA)

        # 3. Turntable rotary bearing ring (silver metallic)
        cv2.rectangle(canvas, (int(bx - 20), int(by - 12)), (int(bx + 20), int(by - 6)), ARM_SERVO_SHAFT, -1)
        cv2.rectangle(canvas, (int(bx - 20), int(by - 12)), (int(bx + 20), int(by - 6)), (120, 125, 130), 1)
        # Green turntable top plate
        cv2.rectangle(canvas, (int(bx - 18), int(by - 15)), (int(bx + 18), int(by - 12)), ARM_LINK_GREEN, -1)
        cv2.rectangle(canvas, (int(bx - 18), int(by - 15)), (int(bx + 18), int(by - 12)), ARM_LINK_BORDER, 1)

        # 4. Shoulder Joint & Base Rotation Servo Block (Black Cube Housing)
        # Shoulder joint pivot point S
        sx, sy = bx, by - 16
        servo_w = 18
        servo_h = 14
        cv2.rectangle(
            canvas,
            (int(sx - servo_w // 2), int(sy - servo_h // 2)),
            (int(sx + servo_w // 2), int(sy + servo_h // 2)),
            ARM_SERVO_BLACK,
            -1,
        )
        cv2.rectangle(
            canvas,
            (int(sx - servo_w // 2), int(sy - servo_h // 2)),
            (int(sx + servo_w // 2), int(sy + servo_h // 2)),
            ARM_SERVO_BORDER,
            1,
        )
        # Green side bracket ears holding the shoulder servo
        cv2.rectangle(canvas, (int(sx - 11), int(sy - 5)), (int(sx - 9), int(sy + 5)), ARM_LINK_GREEN, -1)
        cv2.rectangle(canvas, (int(sx + 9), int(sy - 5)), (int(sx + 11), int(sy + 5)), ARM_LINK_GREEN, -1)
        # Silver pivot shaft
        cv2.circle(canvas, (int(sx), int(sy)), 3, ARM_SERVO_SHAFT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(sx), int(sy)), 1, (20, 20, 20), -1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # D. Shoulder-to-Elbow Structural Link (Solid Mid-Green Beam)
        # ----------------------------------------------------------------------
        dx1 = ex - sx
        dy1 = ey - sy
        len1 = math.hypot(dx1, dy1) or 1.0
        ux1 = dx1 / len1
        uy1 = dy1 / len1
        nx1 = -uy1
        ny1 = ux1
        hw1 = 7.0  # Link width = 14px

        # Solid green link polygon
        p1 = [int(sx + nx1 * hw1), int(sy + ny1 * hw1)]
        p2 = [int(ex + nx1 * hw1), int(ey + ny1 * hw1)]
        p3 = [int(ex - nx1 * hw1), int(ey - ny1 * hw1)]
        p4 = [int(sx - nx1 * hw1), int(sy - ny1 * hw1)]
        pts_link1 = np.array([p1, p2, p3, p4], np.int32)
        cv2.fillPoly(canvas, [pts_link1], ARM_LINK_GREEN, cv2.LINE_AA)
        cv2.polylines(canvas, [pts_link1], True, ARM_LINK_BORDER, 1, cv2.LINE_AA)
        # Machined specular highlight bevel
        cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), ARM_LINK_HILIGHT, 1, cv2.LINE_AA)

        # Longitudinal bracket slot cutout (authentic real JetArm detail)
        slot1_start = (int(sx + dx1 * 0.22), int(sy + dy1 * 0.22))
        slot1_end = (int(sx + dx1 * 0.78), int(sy + dy1 * 0.78))
        cv2.line(canvas, slot1_start, slot1_end, ARM_LINK_SLOT, 2, cv2.LINE_AA)
        # Bracket rivet/screw hole accents
        cv2.circle(canvas, (int(sx + dx1 * 0.16), int(sy + dy1 * 0.16)), 1, (24, 28, 24), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(sx + dx1 * 0.84), int(sy + dy1 * 0.84)), 1, (24, 28, 24), -1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # E. Elbow Joint Servo Motor Housing (Distinct Black Cube/Block)
        # ----------------------------------------------------------------------
        # Black rectangular servo box oriented along upper arm direction
        servo_l2 = 9.5
        servo_w2 = 8.0
        c1 = (int(ex - ux1 * servo_l2 + nx1 * servo_w2), int(ey - uy1 * servo_l2 + ny1 * servo_w2))
        c2 = (int(ex + ux1 * servo_l2 + nx1 * servo_w2), int(ey + uy1 * servo_l2 + ny1 * servo_w2))
        c3 = (int(ex + ux1 * servo_l2 - nx1 * servo_w2), int(ey + uy1 * servo_l2 - ny1 * servo_w2))
        c4 = (int(ex - ux1 * servo_l2 - nx1 * servo_w2), int(ey - uy1 * servo_l2 - ny1 * servo_w2))
        pts_elbow = np.array([c1, c2, c3, c4], np.int32)
        cv2.fillPoly(canvas, [pts_elbow], ARM_SERVO_BLACK, cv2.LINE_AA)
        cv2.polylines(canvas, [pts_elbow], True, ARM_SERVO_BORDER, 1, cv2.LINE_AA)

        # Green bracket wrap flanges at elbow joint
        cv2.line(canvas, c1, (int(c1[0] + ux1 * 4), int(c1[1] + uy1 * 4)), ARM_LINK_GREEN, 2, cv2.LINE_AA)
        cv2.line(canvas, c4, (int(c4[0] + ux1 * 4), int(c4[1] + uy1 * 4)), ARM_LINK_GREEN, 2, cv2.LINE_AA)

        # Metallic pivot shaft / servo horn
        cv2.circle(canvas, (int(ex), int(ey)), 4, ARM_SERVO_SHAFT, -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(ex), int(ey)), 2, (30, 30, 30), -1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # F. Elbow-to-Wrist Forearm Structural Link (Solid Mid-Green Tapered Beam)
        # ----------------------------------------------------------------------
        dx2 = wx - ex
        dy2 = wy - ey
        len2 = math.hypot(dx2, dy2) or 1.0
        ux2 = dx2 / len2
        uy2 = dy2 / len2
        nx2 = -uy2
        ny2 = ux2

        hw2_elbow = 6.5
        hw2_wrist = 5.0

        q1 = [int(ex + nx2 * hw2_elbow), int(ey + ny2 * hw2_elbow)]
        q2 = [int(wx + nx2 * hw2_wrist), int(wy + ny2 * hw2_wrist)]
        q3 = [int(wx - nx2 * hw2_wrist), int(wy - ny2 * hw2_wrist)]
        q4 = [int(ex - nx2 * hw2_elbow), int(ey - ny2 * hw2_elbow)]
        pts_link2 = np.array([q1, q2, q3, q4], np.int32)
        cv2.fillPoly(canvas, [pts_link2], ARM_LINK_GREEN, cv2.LINE_AA)
        cv2.polylines(canvas, [pts_link2], True, ARM_LINK_BORDER, 1, cv2.LINE_AA)
        # Machined specular highlight bevel
        cv2.line(canvas, (q1[0], q1[1]), (q2[0], q2[1]), ARM_LINK_HILIGHT, 1, cv2.LINE_AA)

        # Longitudinal bracket slot cutout along forearm
        slot2_start = (int(ex + dx2 * 0.20), int(ey + dy2 * 0.20))
        slot2_end = (int(ex + dx2 * 0.80), int(ey + dy2 * 0.80))
        cv2.line(canvas, slot2_start, slot2_end, ARM_LINK_SLOT, 2, cv2.LINE_AA)
        cv2.circle(canvas, (int(ex + dx2 * 0.15), int(ey + dy2 * 0.15)), 1, (24, 28, 24), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(ex + dx2 * 0.85), int(ey + dy2 * 0.85)), 1, (24, 28, 24), -1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # G. Wrist Servo Block & Depth Camera Sensor Housing
        # ----------------------------------------------------------------------
        # Black rectangular servo housing at wrist
        w_servo_l = 6.0
        w_servo_w = 7.0
        wc1 = (int(wx - ux2 * w_servo_l + nx2 * w_servo_w), int(wy - uy2 * w_servo_l + ny2 * w_servo_w))
        wc2 = (int(wx + ux2 * w_servo_l + nx2 * w_servo_w), int(wy + uy2 * w_servo_l + ny2 * w_servo_w))
        wc3 = (int(wx + ux2 * w_servo_l - nx2 * w_servo_w), int(wy + uy2 * w_servo_l - ny2 * w_servo_w))
        wc4 = (int(wx - ux2 * w_servo_l - nx2 * w_servo_w), int(wy - uy2 * w_servo_l - ny2 * w_servo_w))
        pts_wrist = np.array([wc1, wc2, wc3, wc4], np.int32)
        cv2.fillPoly(canvas, [pts_wrist], ARM_SERVO_BLACK, cv2.LINE_AA)
        cv2.polylines(canvas, [pts_wrist], True, ARM_SERVO_BORDER, 1, cv2.LINE_AA)

        # Compute smooth blend between home diagonal zigzag posture and operational motion
        hx, hy = float(self.home_pos[0]), float(self.home_pos[1])
        home_dist = math.hypot(wx - hx, wy - hy)
        home_blend = max(0.0, min(1.0, 1.0 - home_dist / 110.0))

        # At home/standby, gripper tilts forward/down-left; in motion it aligns with forearm
        gx = -0.65 * home_blend + ux2 * (1.0 - home_blend)
        gy = 0.76 * home_blend + uy2 * (1.0 - home_blend)
        g_len = math.hypot(gx, gy) or 1.0
        gx, gy = gx / g_len, gy / g_len
        gnx, gny = -gy, gx

        # RealSense-style depth camera module mounted on wrist
        # At home, camera head sits neatly on top of the wrist servo pointing forward
        cam_cx = int(wx + 2.0 * home_blend - ux2 * 6.0 * (1.0 - home_blend) + nx2 * 10.0 * (1.0 - home_blend))
        cam_cy = int(wy - 12.0 * home_blend - uy2 * 6.0 * (1.0 - home_blend) + ny2 * 10.0 * (1.0 - home_blend))
        cam_w = 16.0
        cam_h = 6.0
        # Camera module bounding corners
        cam_ux = 0.98 * home_blend + (-nx2) * (1.0 - home_blend)
        cam_uy = 0.17 * home_blend + (ux2) * (1.0 - home_blend)
        cam_nx = -cam_uy
        cam_ny = cam_ux
        cam_p1 = (int(cam_cx - cam_ux * (cam_w / 2) + cam_nx * (cam_h / 2)), int(cam_cy - cam_uy * (cam_w / 2) + cam_ny * (cam_h / 2)))
        cam_p2 = (int(cam_cx + cam_ux * (cam_w / 2) + cam_nx * (cam_h / 2)), int(cam_cy + cam_uy * (cam_w / 2) + cam_ny * (cam_h / 2)))
        cam_p3 = (int(cam_cx + cam_ux * (cam_w / 2) - cam_nx * (cam_h / 2)), int(cam_cy + cam_uy * (cam_w / 2) - cam_ny * (cam_h / 2)))
        cam_p4 = (int(cam_cx - cam_ux * (cam_w / 2) - cam_nx * (cam_h / 2)), int(cam_cy - cam_uy * (cam_w / 2) - cam_ny * (cam_h / 2)))
        pts_cam = np.array([cam_p1, cam_p2, cam_p3, cam_p4], np.int32)
        cv2.fillPoly(canvas, [pts_cam], (20, 20, 22), cv2.LINE_AA)
        cv2.polylines(canvas, [pts_cam], True, (48, 52, 58), 1, cv2.LINE_AA)
        # Camera dual lens optics dots
        cv2.circle(canvas, (int(cam_cx - cam_ux * 3), int(cam_cy - cam_uy * 3)), 1, (100, 110, 120), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(cam_cx + cam_ux * 3), int(cam_cy + cam_uy * 3)), 1, (100, 110, 120), -1, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # H. Black Two-Finger Straight Parallel Gripper
        # ----------------------------------------------------------------------
        # Parallel sliding motion:
        # At home: splayed open claw position like reference photo (spread = 16px, wide open)
        # When gripped: clamped together (spread = 5px)
        spread = (16.0 * home_blend + 15.0 * (1.0 - home_blend)) - (self.grip_progress * 11.0)
        jaw_len = 18.0
        finger_th = 3.5

        # Cross-slider guide bar
        track_start = (int(wx + gx * 3.0 - gnx * (spread + 3.0)), int(wy + gy * 3.0 - gny * (spread + 3.0)))
        track_end = (int(wx + gx * 3.0 + gnx * (spread + 3.0)), int(wy + gy * 3.0 + gny * (spread + 3.0)))
        cv2.line(canvas, track_start, track_end, ARM_SERVO_BLACK, 3, cv2.LINE_AA)
        cv2.line(canvas, track_start, track_end, ARM_SERVO_BORDER, 1, cv2.LINE_AA)

        # Render Left and Right Straight Parallel Finger Plates
        for side in (-1, 1):
            # Inner edge of straight finger plate
            f_in_root = (wx + gx * 3.0 + gnx * (side * spread), wy + gy * 3.0 + gny * (side * spread))
            f_in_tip = (wx + gx * (3.0 + jaw_len) + gnx * (side * spread), wy + gy * (3.0 + jaw_len) + gny * (side * spread))
            # Outer edge of straight finger plate
            f_out_tip = (wx + gx * (3.0 + jaw_len) + gnx * (side * (spread + finger_th)), wy + gy * (3.0 + jaw_len) + gny * (side * (spread + finger_th)))
            f_out_root = (wx + gx * 3.0 + gnx * (side * (spread + finger_th)), wy + gy * 3.0 + gny * (side * (spread + finger_th)))

            pts_finger = np.array(
                [
                    [int(f_in_root[0]), int(f_in_root[1])],
                    [int(f_in_tip[0]), int(f_in_tip[1])],
                    [int(f_out_tip[0]), int(f_out_tip[1])],
                    [int(f_out_root[0]), int(f_out_root[1])],
                ],
                np.int32,
            )
            # Solid black finger plate
            cv2.fillPoly(canvas, [pts_finger], ARM_GRIPPER_FINGER, cv2.LINE_AA)
            cv2.polylines(canvas, [pts_finger], True, ARM_GRIPPER_BORDER, 1, cv2.LINE_AA)

            # Inward textured rubber grip pad at finger tip
            pad_start = (wx + gx * (3.0 + jaw_len - 5.0) + gnx * (side * spread), wy + gy * (3.0 + jaw_len - 5.0) + gny * (side * spread))
            pad_end = f_in_tip
            pad_in = (f_in_tip[0] - gnx * (side * 1.5), f_in_tip[1] - gny * (side * 1.5))
            pad_in_start = (pad_start[0] - gnx * (side * 1.5), pad_start[1] - gny * (side * 1.5))
            pts_pad = np.array(
                [
                    [int(pad_start[0]), int(pad_start[1])],
                    [int(pad_end[0]), int(pad_end[1])],
                    [int(pad_in[0]), int(pad_in[1])],
                    [int(pad_in_start[0]), int(pad_in_start[1])],
                ],
                np.int32,
            )
            cv2.fillPoly(canvas, [pts_pad], ARM_RUBBER_PAD, cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # I. Carried Object (Attached when clamped)
        # ----------------------------------------------------------------------
        if self.carried_object and self.grip_progress > 0.5:
            c_name = self.carried_object.get("class_name", "Cube")
            c_col = self.carried_object.get("color", UI_ACCENT_GREEN)
            sz = 24
            ox = int(wx - sz // 2 + gx * 11)
            oy = int(wy - sz // 2 + gy * 11)

            patch = self.carried_object.get("patch")
            if patch is not None and patch.size > 0:
                try:
                    p_res = cv2.resize(patch, (sz, sz))
                    y1, y2 = max(0, oy), min(canvas.shape[0], oy + sz)
                    x1, x2 = max(0, ox), min(canvas.shape[1], ox + sz)
                    crop_p = p_res[0 : (y2 - y1), 0 : (x2 - x1)]
                    if crop_p.shape[0] > 0 and crop_p.shape[1] > 0:
                        canvas[y1:y2, x1:x2] = crop_p
                except Exception:
                    cv2.rectangle(canvas, (ox, oy), (ox + sz, oy + sz), c_col, -1)
            else:
                cv2.rectangle(canvas, (ox, oy), (ox + sz, oy + sz), c_col, -1)

            cv2.rectangle(canvas, (ox, oy), (ox + sz, oy + sz), (255, 255, 255), 1)

            # Carried floating badge with clean pill styling
            badge_text = f"{c_name.upper()} -> BIN {self.carried_object.get('bin_id', 1)}"
            draw_hud_badge(
                canvas,
                badge_text,
                (ox - 8, oy - 20),
                bg_color=(14, 18, 26),
                text_color=UI_TEXT_TITLE,
                border_color=c_col,
                font_scale=0.34,
                font_thickness=1,
                padding=(6, 3),
            )

        # ----------------------------------------------------------------------
        # J. Gripper State Badge ([ OPEN ] / [ GRIP ])
        # ----------------------------------------------------------------------
        is_gripped = (self.grip_progress > 0.5)
        grip_badge_text = "GRIP" if is_gripped else "OPEN"
        badge_bg = (30, 42, 54) if is_gripped else (18, 38, 24)
        badge_border = UI_ACCENT_AMBER if is_gripped else UI_ACCENT_GREEN
        badge_text_col = UI_ACCENT_AMBER if is_gripped else UI_ACCENT_GREEN
        draw_hud_badge(
            canvas,
            grip_badge_text,
            (int(wx) - 20, int(wy) - 24),
            bg_color=badge_bg,
            text_color=badge_text_col,
            border_color=badge_border,
            font_scale=0.32,
            font_thickness=1,
            padding=(7, 3),
            radius=8,
        )

        # ----------------------------------------------------------------------
        # K. Arm Telemetry Status (Cleanly integrated without bottom-bar overlap)
        # ----------------------------------------------------------------------
        if self.is_running_sequence and self.current_step_idx < len(self.sequence):
            status_text = self.sequence[self.current_step_idx].get("status", self.state)
            status_col = UI_ACCENT_AMBER
        elif self.state == "TRACK_DRAG":
            status_text = "TRACKING DRAG"
            status_col = UI_ACCENT_CYAN
        else:
            status_text = "IDLE"
            status_col = UI_ACCENT_GREEN

        # Store status string for access by HUD / status panels
        self.current_status_label = status_text
        self.current_status_color = status_col
