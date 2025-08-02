from argparse import ArgumentParser
import json
from logging import getLogger, basicConfig
import os
from shutil import rmtree, copy
import sys
import glob

from tqdm import tqdm
from ultralytics.data.converter import convert_coco

# This assumes your configuration file is named 'config.py'.
# If you have named it 'prepare_config.py', change the import statement below.
from config import (
    TRAIN_SPLIT,
    LABELS,
    LOG_FILENAME,
    LOG_FORMAT,
    LOG_LEVEL,
)


basicConfig(filename=LOG_FILENAME, format=LOG_FORMAT, level=LOG_LEVEL)
logger = getLogger(__name__)


def assert_folder_exists(folder_path: str):
    """Checks if a folder exists on disk."""
    if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
        logger.error(f"Folder '{folder_path}' not found.")
        sys.exit(1)


def load_json(json_path: str) -> dict:
    """ Loads a JSON file from disk."""
    with open(json_path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def copy_images(
    image_filenames: list,
    src_images_path: str,
    dst_images_path: str
):
    """
    Copies a list of images from a source directory into a destination directory.
    This version uses file copying instead of symbolic links to ensure the dataset
    is self-contained and to avoid issues with temporary source directories.
    """
    for filename in tqdm(image_filenames, desc=f"Copying images to {os.path.basename(dst_images_path)}"):
        src_file = os.path.abspath(os.path.join(src_images_path, filename))
        dst_file = os.path.abspath(os.path.join(dst_images_path, filename))

        if os.path.isfile(src_file):
            # Always copy the file to prevent broken symbolic links.
            # This is more robust for environments like Kaggle.
            copy(src_file, dst_file)


def prepare_annotations(src_annotations_path: str, dst_path: str):
    """Converts annotations in Ultralytics' format (YOLO)."""
    # cf. https://docs.ultralytics.com/reference/data/converter/#ultralytics.data.converter.convert_coco
    convert_coco(
        labels_dir=src_annotations_path,
        save_dir=dst_path,
        use_segments=False,
        cls91to80=False
    )
    logger.info(f"Annotations saved to {os.path.join(dst_path, 'labels')}")


def prepare_images(
    src_annotations_path: str,
    src_images_path: str,
    dst_path: str
) -> list[dict]:
    """
    Organizes images in Ultralytics' format and returns from the annotations
    the list of object categories (classes) to detect.
    """
    categories = []

    # Copy images into correct folder
    for label in LABELS:
        src_annotations_path_with_label = os.path.join(
            src_annotations_path,
            f"{label}.json"
        )

        # Skip non-existant labels
        if not os.path.exists(src_annotations_path_with_label):
            logger.warning(f"No annotation file for '{label}' split found. Skipping.")
            continue

        logger.info(f"Processing '{label}' images...")
        dst_images_path = os.path.join(dst_path, "images", label)
        os.makedirs(dst_images_path, exist_ok=True)

        # Get list of images to copy
        json_data = load_json(src_annotations_path_with_label)
        image_filenames = [img["file_name"] for img in json_data["images"]]

        copy_images(image_filenames, src_images_path, dst_images_path)

        # Get detection categories (usually the same across all jsons)
        if not categories:
            categories = json_data.get("categories", [])

    logger.info(f"Images saved to {os.path.join(dst_path, 'images')}")
    return categories


def prepare_configuration_file(categories: list[dict], dst_path: str):
    """Creates Ultralytics' configuration file (data.yaml)."""
    # COCO IDs can be non-sequential, so map them carefully.
    # Ultralytics' format requires a zero-indexed, sequential mapping.
    category_names = [c['name'] for c in sorted(categories, key=lambda x: x['id'])]

    # Create YAML configuration file.
    yaml_file_path = os.path.join(dst_path, "data.yaml")
    with open(yaml_file_path, mode="w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(dst_path)}  # dataset root dir\n")
        f.write("train: images/train  # train images (relative to 'path')\n")
        f.write("val: images/val  # val images (relative to 'path')\n")
        f.write("test: images/test  # test images (optional)\n\n")

        f.write("# Classes\n")
        f.write("names:\n")
        for i, name in enumerate(category_names):
            f.write(f"  {i}: {name}\n")

    logger.info(f"Configuration saved to {yaml_file_path}")


def split_train_to_train_and_val(dst_path: str, train_ratio: float):
    """
    Moves a percentage of files from the 'train' directory to a new 'val'
    directory for both images and labels.
    """
    train_image_path = os.path.join(dst_path, "images", "train")
    val_image_path = os.path.join(dst_path, "images", "val")
    train_label_path = os.path.join(dst_path, "labels", "train")
    val_label_path = os.path.join(dst_path, "labels", "val")

    # Ensure validation directories exist
    os.makedirs(val_image_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)

    # Get all image files (symlinks or copies) from the training directory
    files = list(glob.glob(os.path.join(train_image_path, "*")))
    if not files:
        logger.warning("No files found in the training directory to split.")
        return
        
    # Move a fraction of the files to the validation set
    for file_path in files[int(len(files) * train_ratio):]:
        image_file = os.path.basename(file_path)
        label_file = os.path.splitext(image_file)[0] + ".txt"

        src_image = os.path.join(train_image_path, image_file)
        dst_image = os.path.join(val_image_path, image_file)
        src_label = os.path.join(train_label_path, label_file)
        dst_label = os.path.join(val_label_path, label_file)
        
        # Move the image and its corresponding label file
        if os.path.exists(src_image):
            os.rename(src_image, dst_image)
        if os.path.exists(src_label):
            os.rename(src_label, dst_label)


def main(src_path: str, dst_path: str, force_remove: bool = False):
    "Program's entrypoint."
    # Pre-checks.
    if not src_path or not dst_path:
        logger.error("Source or destination folder is empty.")
        sys.exit(1)

    assert_folder_exists(src_path)

    if os.path.exists(dst_path):
        if force_remove:
            logger.warning(f"Destination folder {dst_path} exists. Removing it.")
            rmtree(dst_path)
        else:
            logger.info(f"Destination folder {dst_path} already exists. Exiting.")
            return

    src_annotations_path = os.path.join(src_path, "annotations")
    assert_folder_exists(src_annotations_path)

    src_images_path = os.path.join(src_path, "images")
    assert_folder_exists(src_images_path)

    # Conversion to Ultralytics' format.
    prepare_annotations(src_annotations_path, dst_path)
    categories = prepare_images(src_annotations_path, src_images_path, dst_path)
    if not categories:
        logger.error("No detection categories (classes) could be loaded from the annotation files. Cannot proceed.")
        sys.exit(1)

    # Generate validation dataset if val.json does not exist.
    if not os.path.exists(os.path.join(src_annotations_path, "val.json")):
        logger.info("val.json not found. Generating validation set from the training set.")
        split_train_to_train_and_val(dst_path, TRAIN_SPLIT)

    prepare_configuration_file(categories, dst_path)


if __name__ == "__main__":
    parser = ArgumentParser("Prepare COCO dataset to Ultralytics format.")
    parser.add_argument(
        "--src", type=os.path.abspath, required=True,
        help="Path to source COCO dataset root directory."
    )
    parser.add_argument(
        "--dst", type=os.path.abspath, required=True,
        help="Path to output Ultralytics dataset root directory."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Automatically erase the destination folder if it exists."
    )
    args = parser.parse_args()

    main(args.src, args.dst, args.force)

    logger.info(f"Script finished successfully.")