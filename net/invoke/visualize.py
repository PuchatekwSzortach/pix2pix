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

    import box
    import tqdm
    import vlogging

    import net.data
    import net.utilities

    config = box.Box(net.utilities.read_yaml(config_path))

    data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.facades_dataset.validation_data_dir,
        batch_size=config.facades_model.batch_size,
        shuffle=True,
        is_source_on_left_side=False
    )

    iterator = iter(data_loader)
    logger = net.utilities.get_logger(path=config.logging_path)

    for _ in tqdm.tqdm(range(4)):

        sources, targets = next(iterator)

        logger.info(vlogging.VisualRecord(title="sources", imgs=sources))
        logger.info(vlogging.VisualRecord(title="targets", imgs=targets))
