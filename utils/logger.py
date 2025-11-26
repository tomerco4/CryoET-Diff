# utils/logger.py

import os
import csv
from typing import Optional


class CSVLogger:
    """
    Minimal CSV logger for tracking losses etc.
    """

    def __init__(self, log_dir: str, filename: str = "train_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, filename)
        self._file_exists = os.path.exists(self.path)

    def log(self, **kwargs):
        """
        Append a row to the CSV with the keys of kwargs as columns.
        """
        write_header = not self._file_exists
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(kwargs.keys()))
            if write_header:
                writer.writeheader()
                self._file_exists = True
            writer.writerow(kwargs)
