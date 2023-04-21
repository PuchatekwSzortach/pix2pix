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
