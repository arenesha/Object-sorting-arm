            # ------------------------------------------------------------------
            # B. Masking: Render ONLY pickup zone in full color; BLACK OUT outside
            # ------------------------------------------------------------------
            canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            canvas[ry:ry + rh, rx:rx + rw] = frame[ry:ry + rh, rx:rx + rw]

            # ------------------------------------------------------------------
            # C. Draw Pickup Zone Frame & Precision Viewfinder Reticle
            # ------------------------------------------------------------------
            # Subtle outer perimeter guide
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (38, 52, 70), 1, cv2.LINE_AA)
            # High-tech camera viewfinder corner brackets
            draw_corner_brackets(canvas, (rx, ry), (rx + rw, ry + rh), UI_ACCENT_CYAN, bracket_len=18, thickness=2)
            # Optical midpoint alignment ticks
            cv2.line(canvas, (rx + rw // 2, ry), (rx + rw // 2, ry + 6), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (rx + rw // 2, ry + rh), (rx + rw // 2, ry + rh - 6), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (rx, ry + rh // 2), (rx + 6, ry + rh // 2), UI_ACCENT_CYAN, 1, cv2.LINE_AA)
            cv2.line(canvas, (rx + rw, ry + rh // 2), (rx + rw - 6, ry + rh // 2), UI_ACCENT_CYAN, 1, cv2.LINE_AA)

            # Center targeting reticle when idle
            if ACTIVE_DETECTION is None and not DRAG_STATE.is_dragging:
                pcx = rx + rw // 2
                pcy = ry + rh // 2
                cv2.circle(canvas, (pcx, pcy), 5, (80, 110, 130), 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx - 10, pcy), (pcx - 4, pcy), (80, 110, 130), 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx + 4, pcy), (pcx + 10, pcy), (80, 110, 130), 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx, pcy - 10), (pcx, pcy - 4), (80, 110, 130), 1, cv2.LINE_AA)
                cv2.line(canvas, (pcx, pcy + 4), (pcx, pcy + 10), (80, 110, 130), 1, cv2.LINE_AA)

            # High-contrast dual header plaques safely positioned ABOVE camera zone
            draw_hud_badge(
                canvas,
                "● AI OPTICAL FEED",
                (rx, ry - 23),
                bg_color=(16, 22, 30),
                text_color=UI_ACCENT_CYAN,
                border_color=(45, 65, 85),
                font_scale=0.33,
                font_thickness=1,
                padding=(8, 3),
            )
            draw_hud_badge(
                canvas,
                f"ROI [{rw}x{rh}]",
                (rx + rw - 78, ry - 23),
                bg_color=(14, 18, 24),
                text_color=UI_TEXT_MUTED,
                border_color=(35, 45, 58),
                font_scale=0.30,
                font_thickness=1,
                padding=(6, 3),
            )

            # ------------------------------------------------------------------
            # D. Draw 3 Polished Bin Zone Cards
            # ------------------------------------------------------------------
            is_flashing = (DRAG_STATE.flash_until > now)
            drag_cx = DRAG_STATE.drag_rect[0] + DRAG_STATE.drag_rect[2] // 2 if DRAG_STATE.is_dragging else -999
            drag_cy = DRAG_STATE.drag_rect[1] + DRAG_STATE.drag_rect[3] // 2 if DRAG_STATE.is_dragging else -999

            for bin_id, bin_info in BIN_ZONES.items():
                bx, by, bw, bh = bin_info["rect"]
                flashing_this_bin = is_flashing and (DRAG_STATE.flash_bin_id == bin_id)
                incoming_this_bin = DRAG_STATE.is_dragging and (bx <= drag_cx <= bx + bw and by <= drag_cy <= by + bh)
                draw_bin_card(canvas, bin_id, bin_info, flashing_this_bin, incoming_this_bin)

            # ------------------------------------------------------------------
            # E. Bottom Footer Panel Backdrop (Pedestal for arm and controls)
            # ------------------------------------------------------------------
            bar_top = 432
            cv2.rectangle(canvas, (0, bar_top), (FRAME_WIDTH, FRAME_HEIGHT), (14, 18, 24), -1)
            cv2.line(canvas, (0, bar_top), (FRAME_WIDTH, bar_top), UI_PANEL_BORDER, 1)

            # ------------------------------------------------------------------
            # F. Vision Detection / Interactive Mouse-Drag Handling
            # ------------------------------------------------------------------
            if not DRAG_STATE.is_dragging:
                # Submit canvas to background worker
                vision_worker.submit_frame(canvas)

                det_result = vision_worker.get_result(max_age_sec=1.0)
                if det_result is not None:
                    bx, by, bw, bh = det_result.bounding_box
                    cx = bx + bw // 2
                    cy = by + bh // 2
                    if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
                        det_result = None

                ACTIVE_DETECTION = det_result

                if det_result is not None:
                    bx, by, bw, bh = det_result.bounding_box
                    c_name = det_result.class_name
                    conf = det_result.confidence
                    bin_id = det_result.bin_id

                    box_color = BIN_ZONES.get(bin_id, {}).get("base_color", UI_ACCENT_GREEN)

                    # 1. Bounding box & corner viewfinder brackets
                    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), box_color, 2)
                    draw_corner_brackets(canvas, (bx, by), (bx + bw, by + bh), box_color, bracket_len=8, thickness=2)

                    # 2. Center reticle '+'
                    cx = bx + bw // 2
                    cy = by + bh // 2
                    cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)

                    # 3. Clean floating label badge clamped within pickup zone
                    label_text = f"{c_name.upper()} {int(conf * 100)}% -> BIN {bin_id}"
                    label_y = max(ry + 4, by - 24)
                    draw_hud_badge(
                        canvas,
                        label_text,
                        (bx, label_y),
                        bg_color=(14, 18, 26),
                        text_color=UI_TEXT_TITLE,
                        border_color=box_color,
                        font_scale=0.36,
                        font_thickness=1,
                        padding=(6, 3),
                    )

                    # Store crop patch for drag animation
                    if 0 <= by < by + bh <= FRAME_HEIGHT and 0 <= bx < bx + bw <= FRAME_WIDTH:
                        DRAG_STATE.drag_patch = frame[by:by + bh, bx:bx + bw].copy()

            else:
                dx, dy, dw, dh = DRAG_STATE.drag_rect
                ox, oy, ow, oh = DRAG_STATE.original_rect

                orig_cx = ox + ow // 2
                orig_cy = oy + oh // 2
                cur_cx = dx + dw // 2
                cur_cy = dy + dh // 2

                # 1. Connector guide line
                cv2.line(canvas, (orig_cx, orig_cy), (cur_cx, cur_cy), UI_ACCENT_CYAN, 1, cv2.LINE_AA)

                # 2. Render dragged image patch
                if DRAG_STATE.drag_patch is not None and DRAG_STATE.drag_patch.size > 0:
                    pw, ph = dw, dh
                    patch = cv2.resize(DRAG_STATE.drag_patch, (pw, ph))
                    y1_t, y2_t = max(0, dy), min(FRAME_HEIGHT, dy + ph)
                    x1_t, x2_t = max(0, dx), min(FRAME_WIDTH, dx + pw)
                    patch_cropped = patch[0:(y2_t - y1_t), 0:(x2_t - x1_t)]
                    if patch_cropped.shape[0] > 0 and patch_cropped.shape[1] > 0:
                        canvas[y1_t:y2_t, x1_t:x2_t] = patch_cropped

                # 3. Glowing dragged box & badge
                draw_rounded_rect(canvas, (dx, dy), (dx + dw, dy + dh), UI_ACCENT_CYAN, thickness=2, radius=4)
                drag_label = f"DRAGGING: {DRAG_STATE.drag_class.upper()} ({int(DRAG_STATE.drag_conf * 100)}%)"
                draw_hud_badge(
                    canvas,
                    drag_label,
                    (dx, max(42, dy - 24)),
                    bg_color=(18, 26, 36),
                    text_color=UI_ACCENT_CYAN,
                    border_color=UI_ACCENT_CYAN,
                    font_scale=0.35,
                    padding=(6, 3),
                )

            # ------------------------------------------------------------------
            # G. Hiwonder JetArm Inverse Kinematics & Animated Simulation
            # ------------------------------------------------------------------
            if JETARM is not None:
                # 1. Automatic Sort on Stable Detection (Debounce)
                if ACTIVE_DETECTION is not None and not DRAG_STATE.is_dragging and not JETARM.is_running_sequence:
                    if ACTIVE_DETECTION.class_name == JETARM.debounce_class:
                        JETARM.debounce_hits += 1
                        if JETARM.debounce_hits >= JETARM.debounce_required and JETARM.auto_sort_enabled:
                            abx, aby, abw, abh = ACTIVE_DETECTION.bounding_box
                            JETARM.trigger_sort(
                                (abx + abw // 2, aby + abh // 2),
                                ACTIVE_DETECTION.bin_id,
                                ACTIVE_DETECTION.class_name,
                                ACTIVE_DETECTION.confidence,
                                DRAG_STATE.drag_patch,
                            )
                            JETARM.debounce_hits = 0
                    else:
                        JETARM.debounce_class = ACTIVE_DETECTION.class_name
                        JETARM.debounce_hits = 1
                else:
                    if not DRAG_STATE.is_dragging and not JETARM.is_running_sequence:
                        JETARM.debounce_hits = max(0, JETARM.debounce_hits - 1)

                # 2. Advance Arm Kinematics & Trajectories
                JETARM.update()

                # 3. Draw Hiwonder JetArm, links, joints, shadow, and gripper
                JETARM.draw(canvas)

            # ------------------------------------------------------------------
            # H. Overlay Notification / Confirmation Banner (Middle corridor)
            # ------------------------------------------------------------------
            if DRAG_STATE.status_until > now and DRAG_STATE.status_message:
                banner_w = 460
                banner_h = 28
                bx = (FRAME_WIDTH - banner_w) // 2
                by = 208
                draw_rounded_rect(
                    canvas,
                    (bx, by),
                    (bx + banner_w, by + banner_h),
                    color=DRAG_STATE.status_color,
                    thickness=1,
                    radius=6,
                    fill_color=(14, 20, 28),
                )
                cv2.putText(
                    canvas,
                    DRAG_STATE.status_message,
                    (bx + 14, by + 19),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    DRAG_STATE.status_color,
                    1,
                    cv2.LINE_AA,
                )

            # ------------------------------------------------------------------
            # I. Top HUD Header Bar & Aligned Interactive Controls
            # ------------------------------------------------------------------
            cv2.rectangle(canvas, (0, 0), (FRAME_WIDTH, 38), UI_BG_DARK, -1)
            cv2.line(canvas, (0, 38), (FRAME_WIDTH, 38), UI_PANEL_BORDER, 1)

            # Left: Title + Online Status Chip
            cv2.putText(
                canvas, "SORT | AI VISION ARM", (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, UI_TEXT_TITLE, 2, cv2.LINE_AA
            )
            draw_hud_badge(
                canvas,
                "LIVE",
                (195, 9),
                bg_color=(14, 38, 22),
                text_color=UI_ACCENT_GREEN,
                border_color=UI_ACCENT_GREEN,
                font_scale=0.30,
                font_thickness=1,
                padding=(5, 2),
                radius=3,
            )

            # Center-Right: Auto-Sort Toggle Button
            abx, aby, abw, abh = AUTOSORT_BTN_RECT
            auto_active = JETARM.auto_sort_enabled if JETARM else True
            if auto_active:
                draw_rounded_rect(canvas, (abx, aby), (abx + abw, aby + abh), color=UI_ACCENT_GREEN, thickness=1, radius=5, fill_color=(18, 48, 28))
                cv2.putText(canvas, "[A] AUTO: ON", (abx + 12, aby + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, UI_ACCENT_GREEN, 1, cv2.LINE_AA)
            else:
                draw_rounded_rect(canvas, (abx, aby), (abx + abw, aby + abh), color=(55, 68, 82), thickness=1, radius=5, fill_color=(20, 26, 34))
                cv2.putText(canvas, "[A] AUTO: OFF", (abx + 10, aby + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, UI_TEXT_MUTED, 1, cv2.LINE_AA)

            # Clickable Demo Sort Button
            dbx, dby, dbw, dbh = DEMO_BTN_RECT
            draw_rounded_rect(canvas, (dbx, dby), (dbx + dbw, dby + dbh), color=UI_ACCENT_AMBER, thickness=1, radius=5, fill_color=(34, 44, 58))
            cv2.putText(canvas, "> DEMO [D]", (dbx + 12, dby + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, UI_ACCENT_AMBER, 1, cv2.LINE_AA)

            # Right: Live FPS pill badge
            draw_hud_badge(
                canvas,
                f"{fps:.1f} FPS",
                (566, 7),
                bg_color=(14, 18, 24),
                text_color=UI_ACCENT_CYAN,
                border_color=UI_PANEL_BORDER,
                font_scale=0.36,
                font_thickness=1,
                padding=(6, 3),
                radius=4,
            )

            # ------------------------------------------------------------------
            # J. Bottom Footer: 2-Row Structured Legend & Machine Controls
            # ------------------------------------------------------------------
            # Row 1 (y = 450): Workspace Color-Coded Legend & Active Arm Status
            r1_y = 450
            cv2.circle(canvas, (18, r1_y - 4), 3, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, "Pickup", (26, r1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, UI_TEXT_BODY, 1, cv2.LINE_AA)

            cv2.circle(canvas, (80, r1_y - 4), 3, UI_ACCENT_RED, -1, cv2.LINE_AA)
            cv2.putText(canvas, "Bin 1", (88, r1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, UI_TEXT_BODY, 1, cv2.LINE_AA)

            cv2.circle(canvas, (132, r1_y - 4), 3, UI_ACCENT_GREEN, -1, cv2.LINE_AA)
            cv2.putText(canvas, "Bin 2", (140, r1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, UI_TEXT_BODY, 1, cv2.LINE_AA)

            cv2.circle(canvas, (184, r1_y - 4), 3, UI_ACCENT_BLUE, -1, cv2.LINE_AA)
            cv2.putText(canvas, "Bin 3", (192, r1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, UI_TEXT_BODY, 1, cv2.LINE_AA)

            # Arm State Badge in Row 1 (positioned cleanly before the arm base)
            arm_stat = JETARM.current_status_label if (JETARM and hasattr(JETARM, "current_status_label")) else "IDLE"
            arm_stat_col = JETARM.current_status_color if (JETARM and hasattr(JETARM, "current_status_color")) else UI_ACCENT_GREEN
            draw_hud_badge(
                canvas,
                f"ARM: {arm_stat}",
                (265, r1_y - 12),
                bg_color=(18, 24, 34),
                text_color=arm_stat_col,
                border_color=arm_stat_col,
                font_scale=0.32,
                padding=(7, 2),
                radius=3,
            )

            # Row 2 (y = 470): Tactile 3D Keyboard Keycaps (x < 360)
            r2_y = 470
            keycaps = [
                ("D", "Demo"),
                ("A", "Auto"),
                ("Drag", "Pick"),
                ("Q", "Quit"),
            ]
            kx = 14
            for key_letter, key_desc in keycaps:
                (kw, _), _ = cv2.getTextSize(key_letter, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
                box_w = kw + 10
                # Keycap body
                draw_rounded_rect(canvas, (kx, r2_y - 12), (kx + box_w, r2_y + 3), color=(58, 72, 90), thickness=1, radius=3, fill_color=(24, 30, 40))
                # Top bevel highlight
                cv2.line(canvas, (kx + 2, r2_y - 11), (kx + box_w - 2, r2_y - 11), (80, 96, 118), 1, cv2.LINE_AA)
                # Key letter
                cv2.putText(canvas, key_letter, (kx + 5, r2_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.30, UI_ACCENT_CYAN, 1, cv2.LINE_AA)
                kx += box_w + 5
                # Description label
                cv2.putText(canvas, key_desc, (kx, r2_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.30, UI_TEXT_MUTED, 1, cv2.LINE_AA)
                (dw, _), _ = cv2.getTextSize(key_desc, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
                kx += dw + 10

            cycles_val = JETARM.cycle_count if JETARM else 0
            draw_hud_badge(
                canvas,
                f"CYCLES: {cycles_val}",
                (FRAME_WIDTH - 98, r2_y - 12),
                bg_color=(18, 22, 30),
                text_color=UI_TEXT_TITLE,
                border_color=UI_PANEL_BORDER,
                font_scale=0.30,
                padding=(6, 2),
                radius=3,
            )