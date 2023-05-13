"""
Module with data IO logic
"""

import glob
import os
import random
import typing

import box
import cv2
import imgaug
import more_itertools

import net.processing


class TwinImagesDataLoader:
    """
    Data loader for dataset in which source and target images are merged into a single image.
    Yields tuples (source images, target images)
    """

    def __init__(
            self, data_directory: str, batch_size: int,
            shuffle: bool, is_source_on_left_side: bool,
            target_size: typing.Tuple[int, int],
            use_augmentations: bool, augmentation_parameters: typing.Union[dict, None]):
        """
        Constructor

        Args:
            data_directory (str): path to data directory
            batch_size (int): batch size
            shuffle (bool): if True, images are shuffled randomly
            is_source_on_left_side (bool): if True, it's assumed that left side of twin image represents source
            and right side represent target, otherwise order is flipped
            target_size: typing.Tuple[int, int]: target height and width for images
            use_augmentations (bool): if True, images augmentation is used when drawing samples
            augmentation_parameters (typing.Union[dict, None]): augmentation parameters or None if use_augmentations
            is False
        """

        self.data_map = box.Box({
            "data_directory": data_directory,
            "batch_size": batch_size,
            "is_source_on_left_side": is_source_on_left_side,
            "target_size": target_size
        })

        self.shuffle = shuffle
        all_twin_images_paths = sorted(glob.glob(pathname=os.path.join(data_directory, "*.jpg")))

        # Prune all twin image paths so we have number of elements that can be cleanly split into
        # number of batches
        target_elements_count = (len(all_twin_images_paths) // self.data_map.batch_size) * self.data_map.batch_size

        self.data_map["twin_images_paths"] = all_twin_images_paths[:target_elements_count]

        self.use_augmentations = use_augmentations

        self.augmentation_pipeline = imgaug.augmenters.Sequential([
            imgaug.augmenters.Fliplr(p=0.5),
            imgaug.augmenters.Resize(augmentation_parameters["resized_image_shape"]),
            imgaug.augmenters.CropToFixedSize(
                width=augmentation_parameters["image_shape"][0],
                height=augmentation_parameters["image_shape"][1]
            )
        ]) if use_augmentations is True else None

    def __len__(self) -> int:
        """
        Get number of images in dataset

        Returns:
            int: number of images in dataset
        """

        return len(self.data_map.twin_images_paths) // self.data_map.batch_size

    def __iter__(self):

        while True:

            if self.shuffle:
                random.shuffle(self.data_map.twin_images_paths)

            for paths_batch in more_itertools.chunked(
                    self.data_map.twin_images_paths, self.data_map.batch_size, strict=True):

                sources = []
                targets = []

                for path in paths_batch:

                    raw_twin_image = cv2.imread(path)

                    # Resize twin image to target size with twice target width, since twin image
                    # contains two images side by side
                    twin_image = cv2.resize(
                        raw_twin_image,
                        (2 * self.data_map.target_size[1], self.data_map.target_size[0]),
                        interpolation=cv2.INTER_CUBIC)

                    half_width = twin_image.shape[1] // 2

                    sources.append(
                        twin_image[:, :half_width]
                    )

                    targets.append(twin_image[:, half_width:])

                if self.use_augmentations is True:

                    sources, targets = self.augmentation_pipeline(
                        images=sources,
                        segmentation_maps=targets
                    )

                batches_pair = (sources, targets) if self.data_map.is_source_on_left_side else (targets, sources)

                yield \
                    net.processing.ImageProcessor.normalize_batch(batches_pair[0]), \
                    net.processing.ImageProcessor.normalize_batch(batches_pair[1])
