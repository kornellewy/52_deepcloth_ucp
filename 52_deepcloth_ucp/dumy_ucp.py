import os
from pathlib import Path

import cv2
import numpy as np

from texture_generation import texture_generation


class DummyUCP:
    """
    Just fill upper cloth shin trexture.
    """

    def __init__(self) -> None:
        self.min_bbox_size = 20

    def predict(self, image: np.ndarray, image_parse: np.ndarray) -> np.ndarray:
        cloth_parse = self.get_target_body_parts_parse(image_parse=image_parse)
        cloth_parse = cloth_parse * 255
        kernel = np.ones((5, 5), np.uint8)
        cloth_parse = cv2.dilate(cloth_parse, kernel, iterations=1)
        skin_image = self.get_skin_image(image=image, image_parse=image_parse)
        image = np.where(cloth_parse == (255, 255, 255), skin_image, image)
        return image

    def get_target_body_parts_parse(self, image_parse: np.ndarray) -> np.ndarray:
        parse_cloth = (
            (image_parse == 5).astype(np.uint8)
            + (image_parse == 6).astype(np.uint8)
            + (image_parse == 7).astype(np.uint8)
        )
        return parse_cloth

    def get_skin_body_parts_parse(self, image_parse: np.ndarray) -> np.ndarray:
        parse_skin = (
            (image_parse == 13).astype(np.uint8)  # face
            + (image_parse == 14).astype(np.uint8)  # Left-arm
            + (image_parse == 15).astype(np.uint8)  # Right-arm
        )
        return parse_skin

    def get_skin_image(self, image: np.ndarray, image_parse: np.ndarray) -> np.ndarray:
        skin_parse = self.get_skin_body_parts_parse(image_parse)
        _, skin_parse = cv2.threshold(skin_parse, 0, 255, cv2.THRESH_BINARY)
        skin_parse = cv2.cvtColor(skin_parse, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            skin_parse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        mask = np.zeros_like(image, dtype=np.uint8)
        if len(contours) != 0:
            cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        skin = np.where(mask == (255, 255, 255), image, (0, 0, 0))
        values, counts = np.unique(skin.reshape(-1, 3), return_counts=True, axis=0)
        values_to_counts_map = {
            count: tuple(value)
            for value, count in zip(values, counts)
            if tuple(value) != (0, 0, 0)
        }
        skin_color_value = values_to_counts_map[max(values_to_counts_map)]
        skin = texture_generation(
            texture_value=skin_color_value, width=image.shape[0], height=image.shape[1]
        )
        return skin


if __name__ == "__main__":
    target_dataset = Path("J:/deepcloth/datasets/viton0022/image")
    target_mask_dataset = Path("J:/deepcloth/datasets/viton0022/image_parse_new")
    output_dataset = Path("test")
    ucp = DummyUCP()

    def load_images(path: str):
        images = []
        valid_images = [".jpeg", ".jpg", ".png"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            images.append(os.path.join(path, f))
        return images

    for image_path in load_images(target_dataset.as_posix()):
        image_name = Path(image_path).name
        parse_mask_path = target_mask_dataset.joinpath(image_name).as_posix()
        new_image_path = output_dataset.joinpath(image_name).as_posix()
        image = cv2.imread(image_path)
        image_parse = cv2.imread(parse_mask_path)
        predicion = ucp.predict(image=image, image_parse=image_parse)
        cv2.imwrite(new_image_path, predicion)
        break
