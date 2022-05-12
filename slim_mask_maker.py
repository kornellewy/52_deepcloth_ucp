from pathlib import Path
import os
import importlib.util
from typ

import torch
import cv2
import numpy as np


def load_images(path: str):
    images = []
    valid_images = [".jpeg", ".jpg", ".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(os.path.join(path, f))
    return images


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
        image = cv2.imread(image_path)
        image_mask = full_body_segmentaion.predict_numpy(image)
        kernel = np.ones((5, 5), np.uint8)
        image_mask = cv2.erode(image_mask, kernel, iterations=5)
        image_mask = cv2.dilate(image_mask, kernel, iterations=5)
        image_mask = cv2.resize(
            image_mask, (image.shape[1], image.shape[0]), cv2.INTER_AREA
        )
        image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)
        results = cv2.addWeighted(image, 0.5, image_mask, 0.5, 0.0)
        image_mask_path = output_dataset.joinpath(Path(image_path).name).as_posix()
        cv2.imwrite(image_mask_path, results)


if __name__ == "__main__":
    target_dataset = Path(
        "J:/deepcloth/datasets/cloth_type_dataset4/bluski_tshirt_krotki_renkaw"
    )
    output_dataset = Path("test")
    add_images_mask_to_images(
        target_dataset=target_dataset, output_dataset=output_dataset
    )
