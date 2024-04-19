"""
Module with visualization commands
"""

import invoke


@invoke.task
def visualize_facades_data(_context, config_path):
    """
    Visualize facades data

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import os

    import box
    import tqdm

    import net.data
    import net.processing
    import net.utilities

    config = box.Box(net.utilities.read_yaml(config_path))

    data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.facades_dataset.validation_data_dir,
        batch_size=config.facades_model.batch_size,
        shuffle=True,
        is_source_on_left_side=False,
        use_augmentations=False,
        augmentation_parameters=None
    )

    iterator = iter(data_loader)

    logger = net.utilities.get_images_logger(
        path=config.logging_path,
        images_directory=os.path.join(os.path.dirname(config.logging_path), "images"),
        images_html_path_prefix="images"
    )

    for _ in tqdm.tqdm(range(4)):

        sources, targets = next(iterator)

        logger.log_images(
            title="sources",
            images=net.processing.ImageProcessor.denormalize_batch(sources)
        )

        logger.log_images(
            title="targets",
            images=net.processing.ImageProcessor.denormalize_batch(targets)
        )


@invoke.task
def facades_model_predictions(_context, config_path):
    """
    Visualize facades model predictions

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import box
    import numpy as np
    import tensorflow as tf
    import tqdm
    import vlogging

    import net.data
    import net.ml
    import net.processing
    import net.utilities

    config = box.Box(net.utilities.read_yaml(config_path))

    test_data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.facades_dataset.test_data_dir,
        batch_size=config.facades_model.batch_size,
        shuffle=True,
        is_source_on_left_side=False,
        target_size=config.facades_model.image_shape[:2],
        use_augmentations=False,
        augmentation_parameters=None
    )

    iterator = iter(test_data_loader)
    logger = net.utilities.get_logger(path=config.logging_path)
    generator = tf.keras.models.load_model(config.facades_model.generator_model_path)

    for _ in tqdm.tqdm(range(4)):

        sources, targets = next(iterator)

        for triplet in zip(sources, generator.predict(sources, verbose=False), targets):

            logger.info(
                vlogging.VisualRecord(
                    title="ground truth, fake target, target",
                    imgs=list(
                        net.processing.ImageProcessor.denormalize_batch(
                            np.array(triplet)
                        )
                    )
                )
            )


@invoke.task
def visualize_maps_data(_context, config_path):
    """
    Visualize maps data

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import os

    import box
    import numpy as np
    import tqdm

    import net.data
    import net.processing
    import net.utilities

    config = box.Box(net.utilities.read_yaml(config_path))

    data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.maps_dataset.validation_data_dir,
        batch_size=config.maps_model.batch_size,
        shuffle=True,
        is_source_on_left_side=True,
        target_size=config.maps_model.image_shape[:2],
        use_augmentations=False,
        augmentation_parameters=None
    )

    iterator = iter(data_loader)

    logger = net.utilities.get_images_logger(
        path=config.logging_path,
        images_directory=os.path.join(os.path.dirname(config.logging_path), "images"),
        images_html_path_prefix="images"
    )

    for _ in tqdm.tqdm(range(16)):

        sources, targets = next(iterator)

        for pair in zip(sources, targets):

            logger.log_images(
                title="source, target",
                images=net.processing.ImageProcessor.denormalize_batch(np.array(pair))
            )


@invoke.task
def maps_model_predictions(_context, config_path):
    """
    Visualize maps model predictions

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import box
    import numpy as np
    import tensorflow as tf
    import tqdm
    import vlogging

    import net.data
    import net.ml
    import net.processing
    import net.utilities

    config = box.Box(net.utilities.read_yaml(config_path))

    test_data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.maps_dataset.validation_data_dir,
        batch_size=config.maps_model.batch_size,
        shuffle=True,
        is_source_on_left_side=True,
        target_size=config.maps_model.image_shape[:2],
        use_augmentations=False,
        augmentation_parameters=None
    )

    iterator = iter(test_data_loader)
    logger = net.utilities.get_logger(path=config.logging_path)
    generator = tf.keras.models.load_model(config.maps_model.generator_model_path)

    for _ in tqdm.tqdm(range(16)):

        sources, targets = next(iterator)

        for triplet in zip(sources, generator.predict(sources, verbose=False), targets):

            logger.info(
                vlogging.VisualRecord(
                    title="source, fake target, target",
                    imgs=list(
                        net.processing.ImageProcessor.denormalize_batch(
                            np.array(triplet)
                        )
                    )
                )
            )
