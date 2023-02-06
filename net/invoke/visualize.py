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

    import net.data
    import net.utilities

    config = box.Box(net.utilities.read_yaml(config_path))

    data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.facades_dataset.validation_data_dir,
        batch_size=config.facades_model.batch_size
    )

    iterator = iter(data_loader)

    for _ in tqdm.tqdm(range(4)):

        sample = next(iterator)
        print(sample)
