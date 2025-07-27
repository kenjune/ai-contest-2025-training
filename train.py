from argparse import ArgumentParser
from datetime import datetime
from logging import getLogger, basicConfig
import os

from ultralytics import YOLO
from config import (
    BATCH_SIZE,
    DEVICE,
    IMAGE_SIZE,
    LOG_FILENAME,
    LOG_FORMAT,
    LOG_LEVEL,
    MODELS,
    N_EPOCHS,
    SCRIPT_DIR,
)


basicConfig(filename=LOG_FILENAME, format=LOG_FORMAT, level=LOG_LEVEL)
logger = getLogger(__name__)


def main(model_path: str, yaml_path: str, save_dir: str):
    # Load a pretrained YOLO11 model.
    model = YOLO(model_path)

    # Train the model.
    # cf. https://docs.ultralytics.com/modes/train/#train-settings
    model.train(
        data=yaml_path,
        project=save_dir,
        name=datetime.now().strftime("train_%Y-%m-%d_%H-%M-%S"),
        pretrained=True,
        epochs=N_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE
    )


if __name__ == "__main__":
    parser = ArgumentParser("Train YOLO11 detection model.")
    parser.add_argument(
        "-c", "--config", type=os.path.abspath, required=True,
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, choices=MODELS,
        help="Type of YOLO11 model to train."
    )
    parser.add_argument(
        "-w", "--weights_dir", type=os.path.abspath,
        default=os.path.join(SCRIPT_DIR, "weights"),
        help="Path to pre-trained weights directory."
    )
    parser.add_argument(
        "-o", "--outputs_dir", type=os.path.abspath,
        default=os.path.join(SCRIPT_DIR, "outputs"),
        help="Path to outputs directory (where to save results)."
    )
    args = parser.parse_args()

    model_path = os.path.join(args.weights_dir, args.model + ".pt")
    main(model_path, args.config, args.outputs_dir)

    logger.info(f"Script finished successfully.")
