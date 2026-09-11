"""
Prototype and test script for rendering the UI exactly matching the user's reference image.
"""

import os
import sys
import math
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jetson")))

WIDTH = 754
HEIGHT = 550

# Fonts
WIN_FONTS = os.environ.get("WINDIR", "C:\\Windows") + "\\Fonts"
FONT_BOLD = os.path.join(WIN_FONTS, "segoeuib.ttf")
FONT_REG = os.path.join(WIN_FONTS, "segoeui.ttf")


def get_font(size: int, bold: bool = False):
    try:
        p = FONT_BOLD if bold else FONT_REG
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    except Exception:
        pass
    return ImageFont.load_default()


def draw_dashed_rounded_rect(draw: ImageDraw.ImageDraw, box, radius, outline, width=1, dash=(4, 3)):
    """Draws a dashed rounded rectangle."""
    x1, y1, x2, y2 = box
    r = radius
    dash_len, gap_len = dash
    for seg_start, seg_end, is_horiz in [
        ((x1 + r, y1), (x2 - r, y1), True),
        ((x2, y1 + r), (x2, y2 - r), False),
        ((x2 - r, y2), (x1 + r, y2), True),
        ((x1, y2 - r), (x1, y1 + r), False),
    ]:
        total_len = abs((seg_end[0] - seg_start[0]) if is_horiz else (seg_end[1] - seg_start[1]))
        dir_sign = 1 if ((seg_end[0] >= seg_start[0]) if is_horiz else (seg_end[1] >= seg_start[1])) else -1
        cur = 0
        while cur < total_len:
            nxt = min(cur + dash_len, total_len)
            if is_horiz:
                p1 = (seg_start[0] + dir_sign * cur, seg_start[1])
                p2 = (seg_start[0] + dir_sign * nxt, seg_start[1])
            else:
                p1 = (seg_start[0], seg_start[1] + dir_sign * cur)
                p2 = (seg_start[0], seg_start[1] + dir_sign * nxt)
            draw.line([p1, p2], fill=outline, width=width)
            cur += dash_len + gap_len


def create_base_reference_ui() -> Image.Image:
    """Creates the high-fidelity pre-rendered UI chrome layer matching the reference image."""
    base = Image.new("RGBA", (WIDTH, HEIGHT), (8, 12, 20, 255))

    # 1. Outer Frame Glow and Border
    glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow)
    g_draw.rounded_rectangle((3, 3, WIDTH - 4, HEIGHT - 4), radius=22, outline=(0, 229, 255, 70), width=3)
    glow = glow.filter(ImageFilter.GaussianBlur(3))
    base = Image.alpha_composite(base, glow)

    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle((3, 3, WIDTH - 4, HEIGHT - 4), radius=22, outline=(14, 38, 58, 220), width=1)

    # 2. Top Bar Header
    # Left: Logo Badge with Robot Arm Icon
    logo_rect = (24, 10, 52, 38)
    draw.rounded_rectangle(logo_rect, radius=8, fill=(10, 28, 44, 255), outline=(0, 229, 255, 200), width=1)

    # Robot arm vector icon inside badge
    draw.rectangle((30, 32, 46, 34), fill=(0, 229, 255, 255))
    draw.line([(35, 32), (38, 22)], fill=(0, 229, 255, 255), width=2)
    draw.line([(38, 22), (44, 16)], fill=(0, 229, 255, 255), width=2)
    draw.line([(44, 16), (41, 14)], fill=(0, 229, 255, 255), width=1)
    draw.line([(44, 16), (47, 14)], fill=(0, 229, 255, 255), width=1)
    draw.ellipse((37, 21, 39, 23), fill=(255, 255, 255, 255))

    # Title: AI VISION ARM
    f_title = get_font(18, bold=True)
    draw.text((64, 13), "AI VISION", fill=(255, 255, 255, 255), font=f_title)
    draw.text((166, 13), "ARM", fill=(0, 229, 255, 255), font=f_title)

    # 3. Bin Cards (3 Across)
    bin_configs = [
        {
            "id": 1, "name": "BIN 1", "tag": "LEFT",
            "rect": (24, 62, 244, 198), "color": (255, 51, 102), "glow_rgba": (255, 51, 102, 100),
        },
        {
            "id": 2, "name": "BIN 2", "tag": "CENTER",
            "rect": (262, 62, 477, 198), "color": (0, 230, 118), "glow_rgba": (0, 230, 118, 100),
        },
        {
            "id": 3, "name": "BIN 3", "tag": "RIGHT",
            "rect": (495, 62, 715, 198), "color": (0, 145, 234), "glow_rgba": (0, 145, 234, 100),
        },
    ]

    for b in bin_configs:
        bx1, by1, bx2, by2 = b["rect"]
        col = b["color"]

        # Card outer glow
        card_glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        cg_draw = ImageDraw.Draw(card_glow)
        cg_draw.rounded_rectangle((bx1, by1, bx2, by2), radius=12, outline=b["glow_rgba"], width=2)
        card_glow = card_glow.filter(ImageFilter.GaussianBlur(3))
        base = Image.alpha_composite(base, card_glow)
        draw = ImageDraw.Draw(base)

        # Card body fill with subtle gradient tint
        draw.rounded_rectangle((bx1, by1, bx2, by2), radius=12, fill=(12, 16, 26, 240), outline=col, width=1)

        # Header: Colored dot + BIN N
        cx_dot = bx1 + 18
        cy_dot = by1 + 18
        draw.ellipse((cx_dot - 7, cy_dot - 7, cx_dot + 7, cy_dot + 7), fill=(col[0], col[1], col[2], 50))
        draw.ellipse((cx_dot - 4, cy_dot - 4, cx_dot + 4, cy_dot + 4), fill=(col[0], col[1], col[2], 255))
        draw.text((bx1 + 30, by1 + 9), b["name"], fill=(255, 255, 255, 255), font=get_font(13, bold=True))

        # Tag Pill on right
        tag_str = b["tag"]
        f_tag = get_font(9, bold=True)
        tw = int(draw.textlength(tag_str, font=f_tag)) if hasattr(draw, "textlength") else 32
        pill_w = tw + 16
        pill_x2 = bx2 - 12
        pill_x1 = pill_x2 - pill_w
        pill_y1 = by1 + 10
        pill_y2 = pill_y1 + 18
        draw.rounded_rectangle((pill_x1, pill_y1, pill_x2, pill_y2), radius=9, fill=(14, 20, 30, 255), outline=col, width=1)
        draw.text((pill_x1 + 8, pill_y1 + 3), tag_str, fill=col, font=f_tag)

        # Inner Box (Drop Dock)
        rx1 = bx1 + 12
        ry1 = by1 + 36
        rx2 = bx2 - 12
        ry2 = by2 - 12
        draw.rounded_rectangle((rx1, ry1, rx2, ry2), radius=8, fill=(7, 10, 18, 255))
        draw_dashed_rounded_rect(draw, (rx1, ry1, rx2, ry2), radius=8, outline=(col[0], col[1], col[2], 120), width=1, dash=(4, 3))

        # Double Chevron
        mid_x = (rx1 + rx2) // 2
        cy1 = ry1 + 20
        draw.line([(mid_x - 10, cy1), (mid_x, cy1 + 8)], fill=(150, 170, 190, 255), width=2)
        draw.line([(mid_x, cy1 + 8), (mid_x + 10, cy1)], fill=(150, 170, 190, 255), width=2)
        cy2 = cy1 + 7
        draw.line([(mid_x - 10, cy2), (mid_x, cy2 + 8)], fill=(150, 170, 190, 255), width=2)
        draw.line([(mid_x, cy2 + 8), (mid_x + 10, cy2)], fill=(150, 170, 190, 255), width=2)

        # Text: DROP DOCK
        f_dock = get_font(12, bold=True)
        dw = int(draw.textlength("DROP DOCK", font=f_dock)) if hasattr(draw, "textlength") else 66
        draw.text((mid_x - dw // 2, ry1 + 42), "DROP DOCK", fill=(255, 255, 255, 255), font=f_dock)

        # Subtitle: Auto Deposit
        f_sub = get_font(9, bold=False)
        sw = int(draw.textlength("Auto Deposit", font=f_sub)) if hasattr(draw, "textlength") else 52
        draw.text((mid_x - sw // 2, ry1 + 59), "Auto Deposit", fill=(110, 125, 145, 255), font=f_sub)

    # 4. Camera Pickup Zone Frame & Header Pills
    px1, py1, px2, py2 = (175, 247, 467, 442)  # w=292, h=195

    # Header pills above-left: 📷 AI OPTICAL FEED
    cam_pill_x1 = px1 - 6
    cam_pill_y1 = 215
    cam_pill_x2 = cam_pill_x1 + 138
    cam_pill_y2 = cam_pill_y1 + 22
    draw.rounded_rectangle((cam_pill_x1, cam_pill_y1, cam_pill_x2, cam_pill_y2), radius=11, fill=(6, 24, 36, 255), outline=(0, 229, 255, 220), width=1)
    cx_cam = cam_pill_x1 + 14
    cy_cam = (cam_pill_y1 + cam_pill_y2) // 2
    draw.rounded_rectangle((cx_cam - 5, cy_cam - 3, cx_cam + 5, cy_cam + 4), radius=2, fill=(0, 229, 255, 255))
    draw.polygon([(cx_cam - 2, cy_cam - 4), (cx_cam + 2, cy_cam - 4), (cx_cam + 1, cy_cam - 3), (cx_cam - 1, cy_cam - 3)], fill=(0, 229, 255, 255))
    draw.ellipse((cx_cam - 2, cy_cam - 1, cx_cam + 2, cy_cam + 3), fill=(6, 24, 36, 255))
    draw.text((cam_pill_x1 + 26, cam_pill_y1 + 4), "AI OPTICAL FEED", fill=(0, 229, 255, 255), font=get_font(9, bold=True))

    # Header pills above-right: ROI (200x180)
    roi_pill_x2 = px2 + 6
    roi_pill_x1 = roi_pill_x2 - 86
    roi_pill_y1 = 215
    roi_pill_y2 = roi_pill_y1 + 22
    draw.rounded_rectangle((roi_pill_x1, roi_pill_y1, roi_pill_x2, roi_pill_y2), radius=11, fill=(12, 18, 28, 255), outline=(35, 50, 70, 255), width=1)
    draw.text((roi_pill_x1 + 10, roi_pill_y1 + 4), "ROI (200x180)", fill=(130, 150, 175, 255), font=get_font(9, bold=False))

    # Viewport cyan border with glow
    cam_glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    cam_g_draw = ImageDraw.Draw(cam_glow)
    cam_g_draw.rounded_rectangle((px1 - 2, py1 - 2, px2 + 2, py2 + 2), radius=12, outline=(0, 229, 255, 120), width=2)
    cam_glow = cam_glow.filter(ImageFilter.GaussianBlur(3))
    base = Image.alpha_composite(base, cam_glow)
    draw = ImageDraw.Draw(base)
    draw.rounded_rectangle((px1 - 2, py1 - 2, px2 + 2, py2 + 2), radius=12, outline=(0, 229, 255, 255), width=1)

    # 5. Concentric Reach Radius Rings (behind arm)
    arm_bx = 565
    arm_by = 450
    for r in (65, 105, 145):
        box = (arm_bx - r, arm_by - r, arm_bx + r, arm_by + r)
        draw.arc(box, start=160, end=340, fill=(14, 38, 62, 160), width=1)

    # 6. Bottom Bar Controls
    # Far Left: Edit Button
    draw.rounded_rectangle((24, 472, 94, 522), radius=25, fill=(14, 22, 34, 255), outline=(28, 44, 66, 255), width=1)
    draw.text((42, 484), "Edit", fill=(255, 255, 255, 255), font=get_font(14, bold=True))

    # Center-Left: Inline Color Legend
    leg_y = 485
    draw.ellipse((106, leg_y + 1, 112, leg_y + 7), fill=(255, 214, 0, 255))
    draw.ellipse((128, leg_y + 1, 134, leg_y + 7), fill=(255, 51, 102, 255))
    draw.text((138, leg_y), "Bin 1", fill=(170, 185, 205, 255), font=get_font(9, bold=False))
    draw.ellipse((190, leg_y + 1, 196, leg_y + 7), fill=(0, 230, 118, 255))
    draw.text((200, leg_y), "Bin 2", fill=(170, 185, 205, 255), font=get_font(9, bold=False))
    draw.ellipse((252, leg_y + 1, 258, leg_y + 7), fill=(0, 145, 234, 255))
    draw.text((262, leg_y), "Bin 3", fill=(170, 185, 205, 255), font=get_font(9, bold=False))

    # Keyboard shortcut badges
    btn_y1 = 506
    btn_y2 = 528
    draw.rounded_rectangle((126, btn_y1, 178, btn_y2), radius=5, fill=(14, 22, 34, 255), outline=(28, 44, 66, 255), width=1)
    draw.text((133, btn_y1 + 4), "A", fill=(0, 229, 255, 255), font=get_font(9, bold=True))
    draw.text((148, btn_y1 + 4), "Auto", fill=(130, 145, 165, 255), font=get_font(9, bold=False))

    draw.rounded_rectangle((186, btn_y1, 244, btn_y2), radius=5, fill=(10, 28, 42, 255), outline=(0, 229, 255, 200), width=1)
    draw.text((204, btn_y1 + 4), "Drag", fill=(0, 229, 255, 255), font=get_font(9, bold=True))

    draw.rounded_rectangle((252, btn_y1, 302, btn_y2), radius=5, fill=(14, 22, 34, 255), outline=(28, 44, 66, 255), width=1)
    draw.text((268, btn_y1 + 4), "Pick", fill=(130, 145, 165, 255), font=get_font(9, bold=False))

    draw.rounded_rectangle((310, btn_y1, 360, btn_y2), radius=5, fill=(14, 22, 34, 255), outline=(28, 44, 66, 255), width=1)
    draw.text((328, btn_y1 + 4), "Quit", fill=(130, 145, 165, 255), font=get_font(9, bold=False))

    # 7. Top Status Pills
    # LIVE Pill
    draw.rounded_rectangle((245, 10, 299, 36), radius=13, fill=(8, 34, 22, 255), outline=(0, 230, 118, 255), width=1)
    draw.ellipse((255, 20, 261, 26), fill=(0, 230, 118, 255))
    draw.text((267, 15), "LIVE", fill=(0, 230, 118, 255), font=get_font(10, bold=True))

    # [A] AUTO: ON Pill
    draw.rounded_rectangle((355, 10, 495, 36), radius=13, fill=(6, 36, 34, 255), outline=(0, 191, 165, 255), width=1)
    # Camera Icon
    cx_a = 368
    cy_a = 23
    draw.rounded_rectangle((cx_a - 5, cy_a - 4, cx_a + 5, cy_a + 4), radius=2, fill=(0, 229, 255, 255))
    draw.polygon([(cx_a - 2, cy_a - 5), (cx_a + 2, cy_a - 5), (cx_a + 1, cy_a - 4), (cx_a - 1, cy_a - 4)], fill=(0, 229, 255, 255))
    draw.ellipse((cx_a - 2, cy_a - 2, cx_a + 2, cy_a + 2), fill=(6, 36, 34, 255))
    draw.text((380, 15), "[A] AUTO: ON", fill=(0, 229, 255, 255), font=get_font(10, bold=True))

    # DEMO [D] Pill
    draw.rounded_rectangle((505, 10, 625, 36), radius=13, fill=(36, 28, 8, 255), outline=(255, 160, 0, 255), width=1)
    # Play Icon
    px_d = 520
    py_d = 23
    draw.polygon([(px_d - 4, py_d - 5), (px_d + 5, py_d), (px_d - 4, py_d + 5)], fill=(255, 160, 0, 255))
    draw.text((532, 15), "DEMO [D]", fill=(255, 160, 0, 255), font=get_font(10, bold=True))

    # 30.3 FPS Pill
    draw.rounded_rectangle((636, 10, 720, 36), radius=13, fill=(12, 22, 34, 255), outline=(0, 229, 255, 180), width=1)
    # Speedometer / gauge icon
    sx = 650
    sy = 23
    draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), outline=(0, 229, 255, 255), width=1)
    draw.line([(sx, sy), (sx + 3, sy - 3)], fill=(0, 229, 255, 255), width=1)
    draw.text((662, 15), "30.3 FPS", fill=(0, 229, 255, 255), font=get_font(10, bold=True))

    # 8. Bottom Center: ARM: IDLE Pill
    draw.rounded_rectangle((314, 475, 437, 513), radius=19, fill=(7, 32, 22, 255), outline=(0, 230, 118, 255), width=1)
    # Arm icon
    ax = 330
    ay = 494
    draw.line([(ax - 6, ay + 6), (ax - 1, ay)], fill=(0, 230, 118, 255), width=2)
    draw.line([(ax - 1, ay), (ax + 5, ay - 6)], fill=(0, 230, 118, 255), width=2)
    draw.ellipse((ax - 2, ay - 1, ax + 2, ay + 3), fill=(0, 230, 118, 255))
    draw.text((358, 486), "ARM: IDLE", fill=(0, 230, 118, 255), font=get_font(11, bold=True))

    return base


def render_full_frame(cam_img_path: str = None) -> np.ndarray:
    """Renders the complete frame matching the reference image."""
    base_img = create_base_reference_ui()
    canvas = np.array(base_img.convert("RGB"))
    # Convert RGB to BGR for OpenCV
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    # 1. Composite camera feed into pickup zone
    px1, py1, px2, py2 = (175, 247, 467, 442)
    pw = px2 - px1
    ph = py2 - py1

    if cam_img_path and os.path.exists(cam_img_path):
        cam_src = cv2.imread(cam_img_path)
        if cam_src is not None:
            cam_res = cv2.resize(cam_src, (pw, ph))
            canvas[py1:py2, px1:px2] = cam_res
    else:
        # Fallback dark tone
        cv2.rectangle(canvas, (px1, py1), (px2, py2), (18, 24, 34), -1)

    # 2. Viewfinder corner brackets
    bl = 14
    for cx_b, cy_b, dx, dy in [
        (px1, py1, 1, 1), (px2, py1, -1, 1),
        (px1, py2, 1, -1), (px2, py2, -1, -1),
    ]:
        cv2.line(canvas, (cx_b, cy_b), (cx_b + dx * bl, cy_b), (255, 229, 0), 2, cv2.LINE_AA)
        cv2.line(canvas, (cx_b, cy_b), (cx_b, cy_b + dy * bl), (255, 229, 0), 2, cv2.LINE_AA)

    # 3. Center Optical Reticle (Precision Crosshair)
    pcx = (px1 + px2) // 2
    pcy = (py1 + py2) // 2
    # Cyan optical circle
    cv2.circle(canvas, (pcx, pcy), 8, (255, 229, 0), 1, cv2.LINE_AA)
    # Crosshair tick lines
    cv2.line(canvas, (pcx - 16, pcy), (pcx - 8, pcy), (255, 229, 0), 1, cv2.LINE_AA)
    cv2.line(canvas, (pcx + 8, pcy), (pcx + 16, pcy), (255, 229, 0), 1, cv2.LINE_AA)
    cv2.line(canvas, (pcx, pcy - 16), (pcx, pcy - 8), (255, 229, 0), 1, cv2.LINE_AA)
    cv2.line(canvas, (pcx, pcy + 8), (pcx, pcy + 16), (255, 229, 0), 1, cv2.LINE_AA)

    # 4. Hiwonder JetArm Rendering at (565, 450)
    bx, by = 565, 450
    sx, sy = 565, 435
    ex, ey = 515, 335
    wx, wy = 560, 240

    # Grounded Base Platform
    # Black chassis
    cv2.rectangle(canvas, (bx - 42, by - 6), (bx + 42, by + 4), (22, 22, 26), -1)
    cv2.rectangle(canvas, (bx - 42, by - 6), (bx + 42, by + 4), (55, 65, 78), 1)
    # Green accent bracket
    cv2.rectangle(canvas, (bx - 36, by - 5), (bx - 4, by + 3), (72, 200, 80), -1)
    # Micro mounting holes
    for hx, hy in [(bx - 26, by - 1), (bx - 14, by - 1)]:
        cv2.circle(canvas, (hx, hy), 1, (20, 24, 20), -1, cv2.LINE_AA)
    # Vents and dual micro-LEDs
    for vx in (bx + 8, bx + 14, bx + 20):
        cv2.line(canvas, (vx, by - 3), (vx, by + 2), (42, 50, 60), 1, cv2.LINE_AA)
    cv2.circle(canvas, (bx + 28, by - 1), 1, (0, 240, 100), -1, cv2.LINE_AA)
    cv2.circle(canvas, (bx + 33, by - 1), 1, (0, 180, 255), -1, cv2.LINE_AA)

    # Rotary bearing ring
    cv2.rectangle(canvas, (bx - 18, by - 12), (bx + 18, by - 6), (180, 185, 195), -1)
    cv2.rectangle(canvas, (bx - 18, by - 12), (bx + 18, by - 6), (110, 120, 130), 1)
    cv2.rectangle(canvas, (bx - 15, by - 15), (bx + 15, by - 12), (72, 200, 80), -1)

    # Shoulder Joint Servo Block
    cv2.rectangle(canvas, (sx - 10, sy - 8), (sx + 10, sy + 8), (22, 22, 26), -1)
    cv2.rectangle(canvas, (sx - 10, sy - 8), (sx + 10, sy + 8), (60, 70, 82), 1)
    cv2.circle(canvas, (sx, sy), 3, (180, 185, 195), -1, cv2.LINE_AA)

    # Link 1: Shoulder -> Elbow (Bright Green Beam)
    dx1, dy1 = ex - sx, ey - sy
    l1 = math.hypot(dx1, dy1) or 1.0
    ux1, uy1 = dx1 / l1, dy1 / l1
    nx1, ny1 = -uy1, ux1
    w1 = 7.0

    p1 = (int(sx + nx1 * w1), int(sy + ny1 * w1))
    p2 = (int(ex + nx1 * w1), int(ey + ny1 * w1))
    p3 = (int(ex - nx1 * w1), int(ey - ny1 * w1))
    p4 = (int(sx - nx1 * w1), int(sy - ny1 * w1))
    pts_l1 = np.array([p1, p2, p3, p4], np.int32)
    cv2.fillPoly(canvas, [pts_l1], (72, 220, 80), cv2.LINE_AA)
    cv2.polylines(canvas, [pts_l1], True, (32, 110, 42), 1, cv2.LINE_AA)
    cv2.line(canvas, p1, p2, (120, 245, 130), 1, cv2.LINE_AA)
    # Bracket slot
    cv2.line(canvas, (int(sx + dx1 * 0.22), int(sy + dy1 * 0.22)), (int(sx + dx1 * 0.78), int(sy + dy1 * 0.78)), (42, 130, 50), 2, cv2.LINE_AA)

    # Elbow Joint Servo Block
    cv2.rectangle(canvas, (ex - 9, ey - 9), (ex + 9, ey + 9), (22, 22, 26), -1)
    cv2.rectangle(canvas, (ex - 9, ey - 9), (ex + 9, ey + 9), (60, 70, 82), 1)
    cv2.circle(canvas, (ex, ey), 3, (180, 185, 195), -1, cv2.LINE_AA)

    # Link 2: Elbow -> Wrist
    dx2, dy2 = wx - ex, wy - ey
    l2 = math.hypot(dx2, dy2) or 1.0
    ux2, uy2 = dx2 / l2, dy2 / l2
    nx2, ny2 = -uy2, ux2
    w2 = 6.0

    p5 = (int(ex + nx2 * w2), int(ey + ny2 * w2))
    p6 = (int(wx + nx2 * w2), int(wy + ny2 * w2))
    p7 = (int(wx - nx2 * w2), int(wy - ny2 * w2))
    p8 = (int(ex - nx2 * w2), int(ey - ny2 * w2))
    pts_l2 = np.array([p5, p6, p7, p8], np.int32)
    cv2.fillPoly(canvas, [pts_l2], (72, 220, 80), cv2.LINE_AA)
    cv2.polylines(canvas, [pts_l2], True, (32, 110, 42), 1, cv2.LINE_AA)
    cv2.line(canvas, p5, p6, (120, 245, 130), 1, cv2.LINE_AA)
    cv2.line(canvas, (int(ex + dx2 * 0.22), int(ey + dy2 * 0.22)), (int(ex + dx2 * 0.78), int(ey + dy2 * 0.78)), (42, 130, 50), 2, cv2.LINE_AA)

    # Wrist Servo Block & RealSense Camera
    cv2.rectangle(canvas, (wx - 8, wy - 8), (wx + 8, wy + 8), (22, 22, 26), -1)
    cv2.rectangle(canvas, (wx - 8, wy - 8), (wx + 8, wy + 8), (60, 70, 82), 1)
    # Camera sensor head
    cv2.rectangle(canvas, (wx - 10, wy - 15), (wx + 10, wy - 8), (20, 20, 22), -1)
    cv2.circle(canvas, (wx - 4, wy - 11), 1, (100, 110, 120), -1, cv2.LINE_AA)
    cv2.circle(canvas, (wx + 4, wy - 11), 1, (100, 110, 120), -1, cv2.LINE_AA)

    # Parallel Gripper Fingers (Open)
    gx, gy = -0.65, 0.76
    g_len = math.hypot(gx, gy)
    gx, gy = gx / g_len, gy / g_len
    gnx, gny = -gy, gx
    spread = 16.0
    jaw_len = 16.0

    # Cross guide
    cv2.line(canvas, (int(wx - gnx * (spread + 2)), int(wy - gny * (spread + 2))), (int(wx + gnx * (spread + 2)), int(wy + gny * (spread + 2))), (55, 65, 75), 2, cv2.LINE_AA)

    for side in (-1, 1):
        f_in_root = (wx + gx * 2.0 + gnx * (side * spread), wy + gy * 2.0 + gny * (side * spread))
        f_in_tip = (wx + gx * (2.0 + jaw_len) + gnx * (side * spread), wy + gy * (2.0 + jaw_len) + gny * (side * spread))
        f_out_tip = (wx + gx * (2.0 + jaw_len) + gnx * (side * (spread + 3.5)), wy + gy * (2.0 + jaw_len) + gny * (side * (spread + 3.5)))
        f_out_root = (wx + gx * 2.0 + gnx * (side * (spread + 3.5)), wy + gy * 2.0 + gny * (side * (spread + 3.5)))
        pts_f = np.array([f_in_root, f_in_tip, f_out_tip, f_out_root], np.int32)
        cv2.fillPoly(canvas, [pts_f], (22, 22, 26), cv2.LINE_AA)
        cv2.polylines(canvas, [pts_f], True, (65, 75, 88), 1, cv2.LINE_AA)
        # Rubber pad
        cv2.line(canvas, (int(f_in_tip[0] - gx * 5), int(f_in_tip[1] - gy * 5)), (int(f_in_tip[0]), int(f_in_tip[1])), (38, 40, 45), 2, cv2.LINE_AA)

    # Gripper OPEN Pill directly above gripper
    # Pill at (wx - 22, wy - 30)
    op_x1, op_y1, op_x2, op_y2 = (wx - 22, wy - 30, wx + 22, wy - 12)
    # Convert back to PIL for clean text/pill or draw with rounded rect
    draw_rounded_rect_cv = lambda c, p1, p2, col, th=1, r=6, fill=None: None
    # Let's draw pill in OpenCV
    cv2.rectangle(canvas, (op_x1 + 6, op_y1), (op_x2 - 6, op_y2), (8, 36, 22), -1)
    cv2.rectangle(canvas, (op_x1, op_y1 + 6), (op_x2, op_y2 - 6), (8, 36, 22), -1)
    cv2.circle(canvas, (op_x1 + 6, op_y1 + 6), 6, (8, 36, 22), -1, cv2.LINE_AA)
    cv2.circle(canvas, (op_x2 - 6, op_y1 + 6), 6, (8, 36, 22), -1, cv2.LINE_AA)
    cv2.circle(canvas, (op_x1 + 6, op_y2 - 6), 6, (8, 36, 22), -1, cv2.LINE_AA)
    cv2.circle(canvas, (op_x2 - 6, op_y2 - 6), 6, (8, 36, 22), -1, cv2.LINE_AA)
    # Outline
    cv2.line(canvas, (op_x1 + 6, op_y1), (op_x2 - 6, op_y1), (0, 230, 118), 1, cv2.LINE_AA)
    cv2.line(canvas, (op_x1 + 6, op_y2), (op_x2 - 6, op_y2), (0, 230, 118), 1, cv2.LINE_AA)
    cv2.line(canvas, (op_x1, op_y1 + 6), (op_x1, op_y2 - 6), (0, 230, 118), 1, cv2.LINE_AA)
    cv2.line(canvas, (op_x2, op_y1 + 6), (op_x2, op_y2 - 6), (0, 230, 118), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (op_x1 + 6, op_y1 + 6), (6, 6), 180, 0, 90, (0, 230, 118), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (op_x2 - 6, op_y1 + 6), (6, 6), 270, 0, 90, (0, 230, 118), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (op_x2 - 6, op_y2 - 6), (6, 6), 0, 0, 90, (0, 230, 118), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (op_x1 + 6, op_y2 - 6), (6, 6), 90, 0, 90, (0, 230, 118), 1, cv2.LINE_AA)
    cv2.putText(canvas, "OPEN", (op_x1 + 9, op_y1 + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 230, 118), 1, cv2.LINE_AA)

    return canvas


if __name__ == "__main__":
    out_dir = r"C:\Users\Admin\.gemini\antigravity-ide\brain\a3014d08-dc63-499b-a951-e088f288dc4d"
    # Use user's cropped camera frame if available
    cam_sample = os.path.join(os.path.dirname(__file__), "sample_images", "rubiks_cube_val.jpg")
    frame = render_full_frame(cam_sample)
    out_path = os.path.join(out_dir, "hardware_jetarm_reference_match.jpg")
    cv2.imwrite(out_path, frame)
    print(f"[*] Successfully rendered match snapshot: {out_path}")

