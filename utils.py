"""
code made by Korneliusz Lewczuk korneliuszlewczuk@gmail.com
sorces: 
"""

import os
from typing import List


def load_images(path: str) -> List[str]:
    images = []
    valid_images = [".jpeg", ".jpg", ".png"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        images.append(os.path.join(path, f))
    return images


def find_all_sub_dirs(main_dir_path: str) -> list:
    return [f.path for f in os.scandir(main_dir_path) if f.is_dir()]
