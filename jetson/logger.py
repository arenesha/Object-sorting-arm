"""Autonomous Object-Sorting Robotic Arm - Event & Telemetry Logger

Provides structured console logging and persistent CSV auditing for every sorting cycle.
Records:
- Timestamp (ISO 8601)
- Detected Object Class
- Destination Bin ID
- Cycle Duration (seconds)
- Result Status (SUCCESS / FAILED)
- Detection Confidence & Extra Metadata
"""

import os
import csv
import logging
from datetime import datetime
from typing import Optional


class SortLogger:
    """Manages console formatting and persistent CSV logging of sorting events."""

    def __init__(self, log_csv_path: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            log_csv_path: Path to CSV output file (defaults to 'sort_history.csv' in workspace).
        """
        if log_csv_path is None:
            # Default to workspace root or local directory
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            self.csv_path = os.path.join(base_dir, "sort_history.csv")
        else:
            self.csv_path = log_csv_path

        self._setup_logger()
        self._init_csv()

    def _setup_logger(self):
        """Configure standard Python logger with clean student-friendly format."""
        self.logger = logging.getLogger("SortSystem")
        self.logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers if re-instantiated
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def _init_csv(self):
        """Ensure CSV file exists with standard column headers."""
        if not os.path.exists(self.csv_path):
            try:
                os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "object_class",
                        "bin_id",
                        "cycle_duration_sec",
                        "status",
                        "confidence",
                        "notes",
                    ])
                self.logger.info(f"Initialized new sort history log at '{self.csv_path}'")
            except Exception as e:
                self.logger.error(f"Failed to initialize CSV log file: {e}")

    def log_sort_event(
        self,
        object_class: str,
        bin_id: int,
        duration_sec: float,
        status: str = "SUCCESS",
        confidence: float = 1.0,
        notes: str = "",
    ):
        """
        Record a completed sorting cycle to both console and CSV.
        
        Args:
            object_class: Name of the classified object (e.g. 'red', 'green', 'blue')
            bin_id: Target bin number (e.g. 1, 2, 3)
            duration_sec: Total time elapsed during the 10-step arm sequence
            status: 'SUCCESS' or 'FAILED'
            confidence: Detection confidence (0.0 to 1.0)
            notes: Optional diagnostic details
        """
        now_str = datetime.now().isoformat()
        console_msg = (
            f"SORT EVENT -> Class: {object_class.upper():<5} | Bin: {bin_id} | "
            f"Duration: {duration_sec:5.2f}s | Status: {status} | Conf: {confidence*100:3.0f}%"
        )

        if status.upper() == "SUCCESS":
            self.logger.info(console_msg)
        else:
            self.logger.error(f"{console_msg} | Notes: {notes}")

        # Append to CSV
        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    now_str,
                    object_class,
                    bin_id,
                    f"{duration_sec:.2f}",
                    status,
                    f"{confidence:.2f}",
                    notes,
                ])
        except Exception as e:
            self.logger.error(f"Failed to write sort event to CSV log: {e}")

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)
