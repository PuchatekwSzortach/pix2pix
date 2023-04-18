"""
Module with model training commands
"""

import invoke


@invoke.task
def train_facades_gan(_context, config_path):
    """
    Train GAN model on facades dataset.
    Generator tries to learn to generate real facades photos from facades segmentations

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import box
    import icecream

    import net.ml
    import net.utilities

    config = box.Box(net.utilities.read_yaml(config_path))

    discriminator_patch_shape = 1, config.facades_model.image_shape[0] // 8, config.facades_model.image_shape[1] // 8

    pix2pix = net.ml.Pix2PixModel(
        discriminator_patch_shape=discriminator_patch_shape,
        batch_size=config.facades_model.batch_size,
    )

    icecream.ic(pix2pix)
