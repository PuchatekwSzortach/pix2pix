"""
Module with logging logic
"""

import logging

import numpy as np
import tensorflow as tf
import vlogging

import net.processing


def log_twin_image_predictions(
        logger: logging.Logger, generator: tf.keras.models.Model,
        twin_image: np.ndarray, title: str):
    """
    Log twin image predictions

    Args:
        logger (vlogging.Logger): logger instance
        generator (tf.keras.Model): generator model
        twin_images (tuple): twin images
        title (str): title
    """

    sources = np.array([twin_image[:, :(twin_image.shape[1] // 2)]])
    targets = np.array([twin_image[:, (twin_image.shape[1] // 2):]])

    denormalized_fake_targets = net.processing.ImageProcessor.denormalize_batch(
        generator.predict(
            net.processing.ImageProcessor.normalize_batch(sources),
            verbose=False)
    )

    logger.info(
        vlogging.VisualRecord(
            title=title,
            imgs=[sources[0], denormalized_fake_targets[0], targets[0]]
        )
    )
