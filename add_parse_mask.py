from pathlib import Path
import os
import importlib.util

import torch
import cv2
import numpy as np

from utils import load_images
from person_body_parts_segmentation import PersonBodyPartsSegmentation


def add_images_mask_to_images(target_dataset: Path, output_dataset: Path) -> None:
    device = torch.device("cpu")
    parse_body_segmentaion = PersonBodyPartsSegmentation(
        device=device, output_shape=None
    )

    images_paths = load_images(target_dataset.as_posix())
    for image_path in images_paths:
        image_mask = parse_body_segmentaion.predict_path(image_path)
        image_mask_path = output_dataset.joinpath(Path(image_path).name).as_posix()
        cv2.imwrite(image_mask_path, image_mask)
        break


if __name__ == "__main__":
    target_dataset = Path(
        "J:/deepcloth/datasets/under_cloth_preciction_dataset/start_images"
    )
    output_dataset = Path(
        "J:/deepcloth/datasets/under_cloth_preciction_dataset/start_images_parse_mask"
    )
    add_images_mask_to_images(
        target_dataset=target_dataset, output_dataset=output_dataset
    )
