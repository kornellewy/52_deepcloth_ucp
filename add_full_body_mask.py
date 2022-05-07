from pathlib import Path
import os
import importlib.util

import torch
import cv2
import numpy as np

from utils import load_images


def add_images_mask_to_images(target_dataset: Path, output_dataset: Path) -> None:
    device = torch.device("cpu")
    spec = importlib.util.spec_from_file_location(
        "module.name",
        "J:/deepcloth/deepcloth_dataset_maker/segmentation/full_body_segmentaion.py",
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    full_body_segmentaion = foo.FullBodySegmentaion(device=device, output_shape=None)

    images_paths = load_images(target_dataset.as_posix())
    for image_path in images_paths:
        image_mask = full_body_segmentaion.predict_path(image_path)
        image_mask_path = output_dataset.joinpath(Path(image_path).name).as_posix()
        cv2.imwrite(image_mask_path, image_mask)


if __name__ == "__main__":
    target_dataset = Path(
        "J:/deepcloth/datasets/under_cloth_preciction_dataset/start_images"
    )
    output_dataset = Path(
        "J:/deepcloth/datasets/under_cloth_preciction_dataset/start_images_mask"
    )
    add_images_mask_to_images(
        target_dataset=target_dataset, output_dataset=output_dataset
    )
