"""
Module with model training commands
"""

import invoke


@invoke.task
def train_facades_gan(_context):
    """
    Train GAN model on facades dataset.
    Generator tries to learn to generate real facades photos from facades segmentations

    Args:
        _context (invoke.Context): context instance
    """

    import icecream
    import numpy as np

    import net.ml

    pix2pix = net.ml.Pix2PixModel()

    data = np.zeros((2, 256, 256, 3))
    icecream.ic(data.shape)

    output = pix2pix.generator.predict(data, verbose=False)
    icecream.ic(output.shape)
