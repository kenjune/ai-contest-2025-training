import os
import torch


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


# =========================
# Logger
# =========================

# CRITICAL: 50
# ERROR: 40
# WARNING: 30
# INFO: 20
# DEBUG: 10
# NOTSET: 0
LOG_LEVEL = 20  # Ignore logging messages which are less severe
if LOG_LEVEL < 20:
    LOG_FORMAT = "%(levelname)s %(filename)s (%(lineno)d) : %(message)s"  # Logging messages string format (for development)
else:
    LOG_FORMAT = "%(levelname)s : %(message)s"  # Logging messages string format (for release)
LOG_FILENAME = None  # Save logging messages to file. None: console


# =========================
# Dataset
# =========================

LABELS = ("train", "val", "test")


# =========================
# Training
# =========================

MODELS = ("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x")
DEVICE = [0,] if torch.cuda.is_available() else "cpu"

N_EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16
TRAIN_SPLIT = 0.8
