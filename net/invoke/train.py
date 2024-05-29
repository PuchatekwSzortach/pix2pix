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

    import os

    import box
    import tensorflow as tf

    import net.data
    import net.ml
    import net.utilities

    config = box.Box(net.utilities.read_yaml(config_path))

    training_data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.facades_dataset.training_and_validation_data_dir,
        batch_size=config.facades_model.batch_size,
        shuffle=True,
        is_source_on_left_side=False,
        target_size=config.facades_model.image_shape[:2],
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

    validation_data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.facades_dataset.test_data_dir,
        batch_size=config.facades_model.batch_size,
        shuffle=True,
        is_source_on_left_side=False,
        target_size=config.facades_model.image_shape[:2],
        use_augmentations=False,
        augmentation_parameters=None
    )

    validation_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(validation_data_loader),
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
        learning_rate=config.facades_model.learning_rate
    )

    pix2pix.compile()

    pix2pix.fit(
        x=training_dataset,
        steps_per_epoch=len(training_data_loader),
        epochs=config.facades_model.epochs,
        callbacks=[
            net.ml.ModelCheckpoint(
                target_model=pix2pix.generator,
                checkpoint_path=config.facades_model.generator_model_path,
                saving_interval_in_steps=500
            ),
            net.ml.VisualizationArchivesBuilderCallback(
                generator=pix2pix.generator,
                data_iterator=iter(validation_dataset),
                output_directory=os.path.dirname(config.logging_path),
                file_name="archive",
                logging_interval=200,
                max_archive_size_in_bytes=100 * 1024 * 1024,
                max_archives_count=10
            )
        ]
    )


@invoke.task
def train_maps_gan(_context, config_path):
    """
    Train GAN model on maps dataset.
    Generator tries to learn to generate google maps photos from satelite images

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import os

    import box
    import numpy as np
    import tensorflow as tf

    import net.data
    import net.ml
    import net.utilities

    np.set_printoptions(suppress=True)

    config = box.Box(net.utilities.read_yaml(config_path))

    training_data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.maps_dataset.training_and_validation_data_dir,
        batch_size=config.maps_model.batch_size,
        shuffle=True,
        is_source_on_left_side=True,
        target_size=config.maps_model.image_shape[:2],
        use_augmentations=True,
        augmentation_parameters=config.maps_model.data_augmentation_parameters
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

    validation_data_loader = net.data.TwinImagesDataLoader(
        data_directory=config.maps_dataset.test_data_dir,
        batch_size=config.maps_model.batch_size,
        shuffle=True,
        is_source_on_left_side=True,
        target_size=config.maps_model.image_shape[:2],
        use_augmentations=False,
        augmentation_parameters=None
    )

    validation_dataset = tf.data.Dataset.from_generator(
        generator=lambda: iter(validation_data_loader),
        output_types=(
            tf.float32,
            tf.float32
        ),
        output_shapes=(
            tf.TensorShape([None, None, None, 3]),
            tf.TensorShape([None, None, None, 3]),
        )
    ).prefetch(32)

    discriminator_patch_shape = \
        1, \
        config.maps_model.data_augmentation_parameters.image_shape[0] // 8, \
        config.maps_model.data_augmentation_parameters.image_shape[1] // 8

    pix2pix = net.ml.Pix2PixModel(
        discriminator_patch_shape=discriminator_patch_shape,
        batch_size=config.maps_model.batch_size,
        learning_rate=config.maps_model.learning_rate
    )

    pix2pix.compile()

    pix2pix.fit(
        x=training_dataset,
        steps_per_epoch=len(training_data_loader),
        epochs=config.maps_model.epochs,
        callbacks=[
            net.ml.ModelCheckpoint(
                target_model=pix2pix.generator,
                checkpoint_path=config.maps_model.generator_model_path,
                saving_interval_in_steps=500
            ),
            net.ml.VisualizationArchivesBuilderCallback(
                generator=pix2pix.generator,
                data_iterator=iter(validation_dataset),
                output_directory=os.path.dirname(config.logging_path),
                file_name="archive",
                logging_interval=200,
                max_archive_size_in_bytes=100 * 1024 * 1024,
                max_archives_count=10
            ),
            net.ml.GANLearningRateSchedulerCallback(
                generator_optimizer=pix2pix.optimizers_map["generator_optimizer"],
                discriminator_opitimizer=pix2pix.optimizers_map["discriminator_optimizer"],
                base_learning_rate=config.maps_model.learning_rate,
                epochs_count=config.maps_model.epochs
            )
        ]
    )
