"""
Module with utilities
"""

import datetime
import glob
import logging
import logging.handlers
import os
import shutil
import typing
import uuid

import cv2
import numpy as np

import yaml


def get_logger(path: str) -> logging.Logger:
    """
    Returns a logger configured to write to a file
    :param path: path to file logger should write to
    :return: logger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("image_retrieval")
    file_handler = logging.FileHandler(path, mode="w")

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger


class CustomRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler with a custom file naming scheme
    """

    def rotation_filename(self, default_name):

        directory = os.path.dirname(default_name)

        # Use scheme: base.number.extension instead of default base.extension.number
        base, extension, number = os.path.basename(default_name).split(".")

        new_base_name = f"{base}.{number}.{extension}"

        return os.path.join(directory, new_base_name)


def read_yaml(path: str):
    """Read content of yaml file from path

    :param path: path to yaml file
    :type path: str
    :return: yaml file content, usually a dictionary
    """

    with open(path, encoding="utf-8") as file:

        return yaml.safe_load(file)


class ImagesLogger(logging.Logger):
    """
    Logger that adds ability to log images to html
    """

    def __init__(self, name: str, images_directory: str, images_html_path_prefix: str) -> None:
        """
        Constructor

        Args:
            name (str): logger's name
            images_directory (str): path to directory in which images should be stored
            images_html_path_prefix (str): prefix for images paths in html
        """
        super().__init__(name)

        self.images_directory = images_directory
        self.images_html_path_prefix = images_html_path_prefix

    def log_images(self, title, images: typing.List[np.array]):
        """
        Log images as html img tags

        Args:
            title (str): title for header placed above images
            images (typing.List[np.array]): list of images to log
        """

        self.info("<h2>{}</h2>".format(title))

        for image in images:

            image_name = f"{datetime.datetime.now().strftime('%Y-%m-%dT%H_%M_%S')}_{uuid.uuid4()}"

            image_path_on_drive = os.path.join(self.images_directory, "{}.jpg".format(image_name))
            image_path_in_html = os.path.join(self.images_html_path_prefix, "{}.jpg".format(image_name))

            cv2.imwrite(image_path_on_drive, image)

            self.info("<img src='{}'>".format(image_path_in_html))

        self.info("<br>")


def get_images_logger(path: str, images_directory: str, images_html_path_prefix: str) -> ImagesLogger:
    """
    Returns logger that has ability to log images to html

    Args:
        path (str): path to log file
        images_directory (str): path to directory in which images should be stored.
        Directory will be cleared of any previous content if it exists
        images_html_path_prefix (str): prefix for images paths in html

    Returns:
        ImagesLogger: configured ImagesLogger instance
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(images_directory):
        shutil.rmtree(images_directory)

    old_logs = glob.glob(path + "*")

    for log in old_logs:
        os.remove(log)

    os.makedirs(images_directory, exist_ok=True)

    logger = ImagesLogger(
        name=path,
        images_directory=images_directory,
        images_html_path_prefix=images_html_path_prefix
    )

    file_handler = CustomRotatingFileHandler(
        path,
        mode="w",
        maxBytes=256 * 1024,
        backupCount=100
    )

    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.propagate = False

    return logger
