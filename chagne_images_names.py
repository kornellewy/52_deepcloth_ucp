from pathlib import Path
import os

from utils import load_images


def change_images_name_in_targer_dir(
    target_dir_path: Path, start_indxe: int = 0
) -> None:
    images_paths = load_images(target_dir_path.as_posix())
    for image_index, image_path in enumerate(images_paths):
        new_image_path = Path(image_path).parent.joinpath(
            f"{start_indxe+image_index}.png"
        )
        os.rename(image_path, new_image_path)


if __name__ == "__main__":
    target_dataset = Path(
        "J:/deepcloth/datasets/under_cloth_preciction_dataset/raw_images"
    )
    change_images_name_in_targer_dir(target_dir_path=target_dataset)
