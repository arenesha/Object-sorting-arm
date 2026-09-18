"""
Unit test for Jetson Arm real-time cursor drag synchronization and post-drop debounce cooldown.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jetson")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard")))

from jetarm_overlay import JetArmOverlay


def test_jetarm_cursor_drag_lifecycle():
    arm = JetArmOverlay(
        base_pos=(485, 466),
        home_pos=(485, 225),
        pickup_pos=(320, 350),
    )

    # 1. Initial Home state
    assert arm.state == "IDLE"
    assert arm.tip_x == 485.0
    assert arm.tip_y == 225.0
    assert not arm.is_running_sequence

    # 2. Mouse drag start
    arm.sync_mouse_drag_start((320, 350), "Rubiks-Cube", 2)
    assert arm.state == "TRACK_DRAG"
    assert arm.is_running_sequence is False
    assert arm.tip_x == 320.0
    assert arm.tip_y == 350.0
    assert arm.grip_progress == 1.0
    assert arm.carried_object is not None
    assert arm.carried_object["class_name"] == "Rubiks-Cube"
    assert arm.carried_object["bin_id"] == 2

    # 3. Mouse drag move across screen
    arm.sync_mouse_drag_update((250, 200))
    assert arm.tip_x == 250.0
    assert arm.tip_y == 200.0

    arm.sync_mouse_drag_update((320, 115))
    assert arm.tip_x == 320.0
    assert arm.tip_y == 115.0

    # 4. Mouse drop into Bin 2
    arm.sync_mouse_drag_drop(2)
    assert arm.state == "RELEASE" or arm.is_running_sequence
    assert arm.carried_object is None
    assert arm.grip_progress == 0.0
    assert arm.post_sort_cooldown > time.time() + 2.0  # Cooldown is 3.0s in the future
    assert arm.is_running_sequence is True
    assert arm.sequence[0]["name"] == "TO_HOME"

    print("[PASS] test_jetarm_cursor_drag_lifecycle passed!")


def test_flask_endpoints():
    import app as dashboard_app
    client = dashboard_app.app.test_client()

    # Ensure jetarm is initialized
    dashboard_app.STATE.init_jetarm()
    arm = dashboard_app.STATE.jetarm

    # Test /api/arm/drag action: start
    res = client.post(
        "/api/arm/drag",
        json={"action": "start", "x": 310, "y": 340, "class_name": "TestCube", "bin_id": 1},
    )
    assert res.status_code == 200
    assert res.json["success"] is True
    assert arm.state == "TRACK_DRAG"
    assert arm.tip_x == 310.0
    assert arm.tip_y == 340.0
    assert arm.carried_object["class_name"] == "TestCube"

    # Test /api/arm/drag action: move
    res = client.post(
        "/api/arm/drag",
        json={"action": "move", "x": 200, "y": 180},
    )
    assert res.status_code == 200
    assert arm.tip_x == 200.0
    assert arm.tip_y == 180.0

    # Test /api/arm/drag action: drop
    res = client.post(
        "/api/arm/drag",
        json={"action": "drop", "x": 110, "y": 115, "dropped_bin": 1},
    )
    assert res.status_code == 200
    assert arm.carried_object is None
    assert arm.post_sort_cooldown > time.time()
    assert arm.is_running_sequence is True
    assert arm.sequence[0]["name"] == "TO_HOME"

    # Test /api/simulated_sort with source: 'manual_drag' does NOT overwrite arm with trigger_sort
    # While returning home, running sequence should remain TO_HOME
    res = client.post(
        "/api/simulated_sort",
        json={"class_name": "TestCube", "bin_id": 1, "confidence": 0.95, "source": "manual_drag"},
    )
    assert res.status_code == 200
    assert arm.sequence[0]["name"] == "TO_HOME"

    # Test /api/simulated_sort with source: 'demo_button' DOES call trigger_sort when arm is ready
    arm.is_running_sequence = False
    res = client.post(
        "/api/simulated_sort",
        json={"class_name": "TestCube", "bin_id": 1, "confidence": 0.95, "source": "demo_button"},
    )
    assert res.status_code == 200
    assert arm.sequence[0]["name"] == "TO_PICK"

    print("[PASS] test_flask_endpoints passed!")


if __name__ == "__main__":
    test_jetarm_cursor_drag_lifecycle()
    test_flask_endpoints()
    print("\nALL CURSOR DRAG & SORT TESTS PASSED PERFECTLY!")
