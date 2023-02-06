"""
Module with data IO logic
"""

import glob
import os


class TwinImagesDataLoader:
    """
    Data loader for dataset in which source and target images are merged into a single image.
    Yields tuples (source images, target images)
    """

    def __init__(self, data_directory: str, batch_size: int):
        """
        Constructor

        Args:
            data_directory (str): path to data directory
            batch_size (int): batch size
        """

        self.data_directory = data_directory
        self.batch_size = batch_size

        self.twin_images_paths = sorted(glob.glob(pathname=os.path.join(data_directory, "*.jpg")))

    def __len__(self) -> int:
        """
        Get number of images in dataset

        Returns:
            int: number of images in dataset
        """

        return len(self.twin_images_paths)

    def __iter__(self):

        while True:
            yield [], []
