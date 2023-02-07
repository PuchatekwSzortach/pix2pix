"""
Module with data IO logic
"""

import glob
import os
import random

import cv2
import more_itertools


class TwinImagesDataLoader:
    """
    Data loader for dataset in which source and target images are merged into a single image.
    Yields tuples (source images, target images)
    """

    def __init__(
            self, data_directory: str, batch_size: int,
            shuffle: bool, is_source_on_left_side: bool):
        """
        Constructor

        Args:
            data_directory (str): path to data directory
            batch_size (int): batch size
            shuffle (bool): if True, images are shuffled randomly
            is_source_on_left_side (bool): if True, it's assumed that left side of twin image represents source
            and right side represent target, otherwise order is flipped
        """

        self.data_directory = data_directory
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_source_on_left_side = is_source_on_left_side

        all_twin_images_paths = sorted(glob.glob(pathname=os.path.join(data_directory, "*.jpg")))

        # Prune all twin image paths so we have number of elements that can be cleanly split into
        # number of batches
        target_elements_count = len(all_twin_images_paths) // self.batch_size
        self.twin_images_paths = all_twin_images_paths[:target_elements_count]

    def __len__(self) -> int:
        """
        Get number of images in dataset

        Returns:
            int: number of images in dataset
        """

        return len(self.twin_images_paths) // self.batch_size

    def __iter__(self):

        while True:

            if self.shuffle:
                random.shuffle(self.twin_images_paths)

            for paths_batch in more_itertools.chunked(self.twin_images_paths, self.batch_size, strict=True):

                sources = []
                targets = []

                for path in paths_batch:

                    twin_image = cv2.imread(path)

                    half_width = twin_image.shape[1] // 2

                    sources.append(
                        twin_image[:, :half_width]
                    )

                    targets.append(twin_image[:, half_width:])

                yield (sources, targets) if self.is_source_on_left_side else (targets, sources)
