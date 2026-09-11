"""Autonomous Object-Sorting Robotic Arm - YOLO Model Selector CLI

Utility to discover, inspect, and select active YOLO models (.pt / .engine) in /jetson/models/.
Automatically inspects model classes via ultralytics metadata and synchronizes
the class-to-bin mapping JSON file.

Usage:
    python select_model.py                 # Interactive selection menu
    python select_model.py --list          # List all available models and current active
    python select_model.py --set <file>    # Set active model by filename
    python select_model.py --show-mapping  # Display current class-to-bin mapping
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Add jetson module directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import (
    MODELS_DIR,
    MODELS_CONFIG_FILE,
    get_available_models,
    get_active_model_info,
)


def format_size(bytes_size: int) -> str:
    """Format file size in human-readable units."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def list_models():
    """Print all available models in models directory with status."""
    models = get_available_models()
    active_name, _ = get_active_model_info()

    print("=" * 70)
    print("                 YOLO MODEL REPOSITORY & STATUS                   ")
    print(f"Directory: {MODELS_DIR}")
    print("=" * 70)

    if not models:
        print(f"No .pt or .engine models found in '{MODELS_DIR}'.")
        print("Please drop your trained weights into this folder.\n")
        return []

    print(f"{'#':<3} | {'Model Filename':<30} | {'Size':<10} | {'Status':<10}")
    print("-" * 70)
    for idx, m_file in enumerate(models, 1):
        full_p = os.path.join(MODELS_DIR, m_file)
        size_str = format_size(os.path.getsize(full_p))
        status = "[ACTIVE]" if m_file == active_name else ""
        print(f"[{idx}] | {m_file:<30} | {size_str:<10} | {status:<10}")

    print("=" * 70)
    return models


def inspect_model_classes(model_filename: str):
    """
    Attempt to extract class names from model metadata, auto-generating
    <model_name>_mapping.json if needed.
    """
    full_path = os.path.join(MODELS_DIR, model_filename)
    model_stem = os.path.splitext(model_filename)[0]
    mapping_path = os.path.join(MODELS_DIR, f"{model_stem}_mapping.json")

    class_names = []

    try:
        from ultralytics import YOLO
        model = YOLO(full_path)
        raw_names = model.names
        if isinstance(raw_names, dict):
            class_names = [str(raw_names[k]) for k in sorted(raw_names.keys())]
        elif isinstance(raw_names, (list, tuple)):
            class_names = [str(n) for n in raw_names]
        else:
            class_names = [str(raw_names)]
    except Exception as e:
        # Fallback if offline/dummy model
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, "r", encoding="utf-8") as f:
                    class_names = list(json.load(f).keys())
            except Exception:
                pass
        if not class_names:
            class_names = ["class_0", "class_1", "class_2"]

    # Sync mapping file
    existing = {}
    file_existed = os.path.exists(mapping_path)
    if file_existed:
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    updated = dict(existing)
    for c in class_names:
        if c not in updated:
            updated[c] = 0

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)

    return mapping_path, updated


def display_mapping(model_filename: str):
    """Print the class-to-bin mapping table for the given model."""
    mapping_path, mapping = inspect_model_classes(model_filename)

    print(f"\nClass-to-Bin Mapping for '{model_filename}':")
    print(f"Mapping File: {mapping_path}")
    print("-" * 55)
    print(f"{'Class Name':<30} | {'Assigned Bin':<15}")
    print("-" * 55)

    unassigned = []
    for cls_name, b_id in mapping.items():
        bin_str = f"Bin {b_id}" if b_id > 0 else "0 (SKIPPED)"
        print(f"{cls_name:<30} | {bin_str:<15}")
        if b_id == 0:
            unassigned.append(cls_name)

    print("-" * 55)

    if unassigned:
        print("\n>> ATTENTION: The following classes are currently unassigned (bin 0):")
        print(f"   {unassigned}")
        print(f">> Edit '{mapping_path}' to set target bin numbers (1, 2, 3...) before running sort.\n")
    else:
        print("\nAll classes are mapped to active bins!\n")


def set_active_model(model_filename: str):
    """Save active model selection to models_config.json and display its mapping."""
    available = get_available_models()
    if model_filename not in available:
        print(f"Error: Model '{model_filename}' not found in '{MODELS_DIR}'.")
        print(f"Available models: {available}")
        return False

    cfg_data = {
        "active_model": model_filename,
        "selected_at": datetime.now().isoformat(),
        "model_path": os.path.join(MODELS_DIR, model_filename),
    }

    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(MODELS_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)
        print(f"\n>> SUCCESS: Active model set to: '{model_filename}'")
        print(f">> Configuration saved to: '{MODELS_CONFIG_FILE}'")

        display_mapping(model_filename)
        return True
    except Exception as e:
        print(f"Error saving model configuration: {e}")
        return False


def interactive_select():
    """Interactive console prompt to select a model."""
    models = list_models()
    if not models:
        return

    print("\nEnter the number of the model you want to activate (or '0' to cancel):")
    choice = input("Choice: ").strip()

    try:
        val = int(choice)
        if val == 0:
            print("Cancelled.")
            return
        if 1 <= val <= len(models):
            selected = models[val - 1]
            set_active_model(selected)
        else:
            print("Invalid index.")
    except ValueError:
        print("Invalid input.")


def main():
    parser = argparse.ArgumentParser(description="YOLO Model Selector and Mapping Configurator")
    parser.add_argument("--list", action="store_true", help="List available models and current active")
    parser.add_argument("--set", type=str, help="Set active model filename (e.g. --set my_model.pt)")
    parser.add_argument("--show-mapping", action="store_true", help="Show class-to-bin mapping for current active model")
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.set:
        set_active_model(args.set)
        return

    if args.show_mapping:
        active_name, _ = get_active_model_info()
        if active_name:
            display_mapping(active_name)
        else:
            print("No active model currently configured.")
        return

    # Default: interactive mode
    interactive_select()


if __name__ == "__main__":
    main()
