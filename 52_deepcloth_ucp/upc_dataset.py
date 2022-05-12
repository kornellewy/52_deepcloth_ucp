import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations as A

from utils.utils import load_images


class UPCDataset(data.Dataset):
    def __init__(
        self,
        dataset_path,
        images_shape=(192, 256),
        dirs_names={
            "image_dir_name": "image",
            "pose_dir_name": "pose",
            "image_mask": "image_mask",
        },
    ):
        super().__init__()
        self.images_shape = images_shape
        self.dirs_names = dirs_names
        self.dataset_path = dataset_path
        self.dirs_paths = {
            "image_dir_path": str(
                Path(self.dataset_path) / self.dirs_names["image_dir_name"]
            ),
            "image_parse_dir_path": str(
                Path(self.dataset_path) / self.dirs_names["image_parse_dir_name"]
            ),
            "image_mask_dir_path": str(
                Path(self.dataset_path) / self.dirs_names["image_mask"]
            ),
        }
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.transform_mask = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )
        self.images_paths = sorted(list(load_images(self.dirs_paths["image_dir_path"])))
        self.agumentations = A.Compose(
            [
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Cutout(
                    num_holes=16,
                    max_h_size=32,
                    max_w_size=32,
                    fill_value=0,
                    always_apply=True,
                ),
            ],
            keypoint_params=A.KeypointParams(format="xy"),
            additional_targets={
                "keypoints": "keypoints",
                "image_mask": "mask",
            },
        )

    def __len__(self) -> int:
        return len(self.images_paths)

    def __getitem__(self, index: int) -> dict:
        (
            image_name,
            image_path,
            mask_path,
            pose_points_path,
        ) = self.get_items_paths(index=index)
        image, mask = self.read_images_paths(image_path=image_path, mask_path=mask_path)
        pose_data = self.read_pose_points(pose_points_path=pose_points_path)
        image, mask, pose_data = self.agument_data(
            image=image, mask=mask, pose_data=pose_data
        )
        pose_map = self.create_pose_map(pose_data=pose_data)
        image, mask = self.transform(image=image, mask=mask)
        pose_map = self.transform_stack_of_gray_images(pose_map)
        return {
            "image_name": image_name,
            "image": image,
            "mask": mask,
            "pose_map": pose_map,
        }

    def get_items_paths(self, index: int) -> Tuple[str, ...]:
        image_path = self.images_paths[index]
        image_name = Path(image_path).name
        mask_path = (
            Path(self.dirs_paths["image_mask_dir_path"]) / image_name
        ).as_posix()
        if image_name.endswith(".jpg"):
            pose_points_name = image_name.replace(".jpg", "_keypoints.json")
        else:
            pose_points_name = image_name.replace(".png", "_keypoints.json")
        pose_points_path = (
            Path(self.dirs_paths["pose_dir_path"]) / pose_points_name
        ).as_posix()
        return image_name, image_path, mask_path, pose_points_path

    @staticmethod
    def read_images_paths(
        image_path: str, mask_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, mask

    def read_pose_points(self, pose_points_path: str) -> np.ndarray:
        with open(pose_points_path, "r") as file:
            pose_label = json.load(file)
            pose_data = pose_label["people"][0]["pose_keypoints"]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
            pose_data = np.delete(pose_data, 2, axis=1)
            pose_data = np.where(pose_data == 192, 192 - 0.01, pose_data)
            pose_data = np.where(pose_data == 256, 256 - 0.01, pose_data)
        pose_data = pose_data.astype(np.uint8)
        return pose_data

    def agument_data(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        pose_data: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        agumented_data = self.agumentations(
            image=image,
            mask=mask,
            keypoints=pose_data,
        )
        image = agumented_data["image"].astype(np.uint8)
        mask = agumented_data["mask"].astype(np.uint8)
        pose_data = np.array(agumented_data["keypoints"])
        return image, mask, pose_data

    def create_pose_map(self, pose_data: np.ndarray) -> np.ndarray:
        point_numers = pose_data.shape[0]
        point_pose_map = np.zeros(
            shape=(18, self.images_shape[1], self.images_shape[0])
        )
        for i_point_num in range(18):
            single_point_map = np.zeros(
                shape=(self.images_shape[1], self.images_shape[0])
            )
            if i_point_num <= point_numers - 1:
                point_x = pose_data[i_point_num, 0]
                point_y = pose_data[i_point_num, 1]
                if point_x > 1 and point_y > 1:
                    single_point_map = cv2.circle(
                        single_point_map,
                        (point_x, point_y),
                        radius=5,
                        color=(255),
                        thickness=-1,
                    )
            point_pose_map[i_point_num, :, :] = single_point_map
        return point_pose_map

    def transform(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        image = self.transform(image)
        mask = self.transform_mask(mask)
        return (
            image,
            mask,
        )

    def transform_stack_of_gray_images(
        self, stack_of_gray_images: np.ndarray
    ) -> torch.tensor:
        gray_images_number = stack_of_gray_images.shape[0]
        stack = torch.zeros(
            gray_images_number, self.images_shape[1], self.images_shape[0]
        )
        for i in range(0, gray_images_number):
            single_gray_image = stack_of_gray_images[i, :, :]
            if np.max(single_gray_image) > 1:
                single_gray_image = np.clip(single_gray_image, a_min=0, a_max=1)
            single_gray_image = self.transform_mask(single_gray_image)
            stack[i, :, :] = single_gray_image
        return stack
