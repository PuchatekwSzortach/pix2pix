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
    import tensorflow as tf

    import net.data
    import net.ml
    import net.utilities

    config = box.Box(net.utilities.read_yaml(config_path))

    training_data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.facades_dataset.training_data_dir,
        batch_size=config.facades_model.batch_size,
        shuffle=True,
        is_source_on_left_side=False,
        use_augmentations=True,
        augmentation_parameters=config.facades_model.data_augmentation_parameters
    )

    training_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(training_data_loader),
        output_types=(
            tf.float32,
            tf.float32
        ),
        output_shapes=(
            tf.TensorShape([None, None, None, 3]),
            tf.TensorShape([None, None, None, 3]),
        )
    ).prefetch(32)

    discriminator_patch_shape = 1, config.facades_model.image_shape[0] // 8, config.facades_model.image_shape[1] // 8

    pix2pix = net.ml.Pix2PixModel(
        discriminator_patch_shape=discriminator_patch_shape,
        batch_size=config.facades_model.batch_size,
    )

    pix2pix.compile()

    pix2pix.fit(
        x=training_dataset,
        steps_per_epoch=len(training_data_loader),
        epochs=config.facades_model.epochs,
    )
