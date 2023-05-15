"""
Module with machine training logic
"""

import glob
import os
import shutil
import tarfile
import tempfile
import typing

import numpy as np
import tensorflow as tf

import net.processing
import net.utilities


class GeneratorBuilder:
    """
    Class for building various pieces of pix2pix generator
    """

    def get_innermost_block(self, input_channels, filters) -> tf.keras.Model:
        """
        Function to build innermost block model
        """

        input_op = tf.keras.layers.Input(shape=(None, None, input_channels))

        x = self.downscale_block(
            input_op=input_op,
            filters=filters,
            use_normalization=False)

        x = self.upscale_block(
            input_op=x,
            filters=input_channels,
            use_dropout=False)

        output_op = tf.keras.layers.Concatenate()([x, input_op])

        return tf.keras.Model(inputs=input_op, outputs=output_op)

    def get_outermost_block(self, submodule: tf.keras.Model) -> tf.keras.Model:
        """
        Function to build outermost block model
        """

        input_op = tf.keras.layers.Input(shape=(None, None, 3))

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), padding="same")(input_op)
        x = submodule(x)
        x = tf.keras.layers.ReLU()(x)

        # Output is same as input
        output_op = tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            activation="tanh")(x)

        return tf.keras.models.Model(input_op, output_op)

    def get_intermediate_block(
            self, input_channels, filters, use_dropout, submodule: tf.keras.Model) -> tf.keras.Model:
        """
        Function to build intermediate block model

        Returns:
            tf.keras.Model: keras model for the block
        """

        input_op = tf.keras.layers.Input(shape=(None, None, input_channels))

        x = self.downscale_block(
            input_op=input_op,
            filters=filters,
            use_normalization=True)

        x = submodule(x)

        upscale_op = self.upscale_block(
            input_op=x,
            filters=input_channels,
            use_dropout=use_dropout)

        output_op = tf.keras.layers.Concatenate()([upscale_op, input_op])

        return tf.keras.Model(inputs=input_op, outputs=output_op)

    def downscale_block(self, input_op, filters: int, use_normalization: bool):
        """
        Downscale block
        """

        x = tf.keras.layers.LeakyReLU(alpha=0.2)(input_op)
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=(2, 2), padding="same")(x)

        if use_normalization is True:
            x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)

        return x

    def upscale_block(self, input_op, filters: int, use_dropout: bool):
        """
        Upscale block
        """

        x = tf.keras.layers.ReLU()(input_op)

        x = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=4,
            strides=(2, 2),
            padding="same")(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)

        if use_dropout is True:
            x = tf.keras.layers.Dropout(rate=0.5)(x)

        return x


class Pix2PixModel(tf.keras.Model):
    """
    Pix2Pix model
    """

    def __init__(self, discriminator_patch_shape: typing.Tuple[int], batch_size: int) -> None:
        """
        Constructor

        Args:
            discriminator_patch_shape (typing.Tuple[int]): expected shape of discriminator output for target data
            batch_size (int): batch size
        """

        super().__init__()

        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()

        self.generator_loss_op = self._get_generator_loss_op(
            discriminator=self.discriminator,
            generator=self.generator,
            patch_shape=discriminator_patch_shape,
            batch_size=batch_size
        )

        self.discriminator_loss_op = self._get_discriminator_loss_op(
            discriminator=self.discriminator,
            generator=self.generator,
            patch_shape=discriminator_patch_shape,
            batch_size=batch_size
        )

        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    def call(self, *args, **kwargs):
        """
        We don't actually need this, but we need to implement it to make Keras happy
        """
        raise NotImplementedError()

    def _get_generator(self) -> tf.keras.Model:
        """
        Get generator model

        Returns:
            tf.keras.Model: generator model
        """

        generator_builder = GeneratorBuilder()

        blocks = {
            8: generator_builder.get_innermost_block(input_channels=512, filters=512)
        }

        blocks[7] = generator_builder.get_intermediate_block(
            input_channels=512,
            filters=512,
            use_dropout=True,
            submodule=blocks[8]
        )

        blocks[6] = generator_builder.get_intermediate_block(
            input_channels=512,
            filters=512,
            use_dropout=True,
            submodule=blocks[7]
        )

        blocks[5] = generator_builder.get_intermediate_block(
            input_channels=256,
            filters=512,
            use_dropout=True,
            submodule=blocks[6]
        )

        blocks[4] = generator_builder.get_intermediate_block(
            input_channels=128,
            filters=256,
            use_dropout=False,
            submodule=blocks[5]
        )

        blocks[3] = generator_builder.get_intermediate_block(
            input_channels=64,
            filters=128,
            use_dropout=False,
            submodule=blocks[4]
        )

        blocks[2] = generator_builder.get_intermediate_block(
            input_channels=64,
            filters=64,
            use_dropout=False,
            submodule=blocks[3]
        )

        outermost_block = generator_builder.get_outermost_block(submodule=blocks[2])

        outermost_block.compile()
        return outermost_block

    def _get_discriminator(self) -> tf.keras.Model:
        """
        Get pixp2pix discriminator model
        """

        def get_discriminator_block(input_op, filters: int, stride: int, use_normalization: bool):
            """
            Get transformation for single discriminator block
            """

            x = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=4, strides=(stride, stride), padding="same")(input_op)

            if use_normalization is True:
                x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)

            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            return x

        image_shape = None, None, 3

        source_image_input_op = tf.keras.layers.Input(image_shape)
        target_image_input_op = tf.keras.layers.Input(image_shape)

        combined_images_op = tf.keras.layers.Concatenate(axis=-1)([source_image_input_op, target_image_input_op])

        x = get_discriminator_block(input_op=combined_images_op, filters=64, stride=2, use_normalization=False)
        x = get_discriminator_block(input_op=x, filters=128, stride=2, use_normalization=True)
        x = get_discriminator_block(input_op=x, filters=256, stride=2, use_normalization=True)
        x = get_discriminator_block(input_op=x, filters=512, stride=1, use_normalization=True)

        output_op = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=(1, 1), padding="same")(x)

        return tf.keras.models.Model([source_image_input_op, target_image_input_op], output_op)

    def _get_generator_loss_op(
            self, discriminator: tf.keras.Model, generator: tf.keras.Model,
            patch_shape: typing.Tuple[int], batch_size: int) -> tf.Tensor:
        """
        Get pix2pix generator loss operation

        Args:
            discriminator (tf.keras.Model): pix2pix discriminator
            generator (tf.keras.Model): pix2pix generator
            patch_shape (typing.Tuple[int]): shape of discriminator's output for data on which model is to be trained
            batch_size (int): batch size

        Returns:
            tf.Tensor: generator loss operation
        """

        discriminator_loss_op = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        image_condition_loss_op = tf.keras.losses.MeanAbsoluteError()

        all_ones_patch = tf.repeat(tf.ones(patch_shape, dtype=tf.float32), repeats=batch_size, axis=0)

        @tf.function
        def loss_op(source_images, target_images):
            """
            Generator loss op

            Args:
                source_images (tf.Tensor): tensor with source images
                target_images (tf.Tensor): target with target images

            Returns:
                tf.Tensor: scalar loss
            """

            generated_images = generator(source_images, training=True)

            discriminator_predictions = discriminator(
                [source_images, generated_images],
                training=False)

            discriminator_fooling_loss = discriminator_loss_op(all_ones_patch, discriminator_predictions)

            image_similarity_loss = image_condition_loss_op(target_images, generated_images)

            return discriminator_fooling_loss + (100.0 * image_similarity_loss)

        return loss_op

    def _get_discriminator_loss_op(
            self, discriminator: tf.keras.Model, generator: tf.keras.Model,
            patch_shape: typing.Tuple[int], batch_size: int) -> tf.Tensor:
        """
        Get pix2pix discriminator loss operation

        Args:
            discriminator (tf.keras.Model): pix2pix discriminator
            generator (tf.keras.Model): pix2pix generator
            patch_shape (typing.Tuple[int]): shape of discriminator's output for data on which model is to be trained
            batch_size (int): batch size

        Returns:
            tf.Tensor: discriminator loss operation
        """

        all_ones_patch = tf.repeat(tf.ones(patch_shape, dtype=tf.float32), repeats=batch_size, axis=0)
        all_zeros_patch = tf.repeat(tf.zeros(patch_shape, dtype=tf.float32), repeats=batch_size, axis=0)

        labels = tf.concat(
            [
                all_ones_patch,
                all_zeros_patch
            ], axis=0
        )

        base_loss_op = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        @tf.function
        def loss_op(source_images, target_images):
            """
            Discriminator loss op

            Args:
                source_images (tf.Tensor): tensor with source images
                target_images (tf.Tensor): target with target images

            Returns:
                tf.Tensor: scalar loss
            """

            generated_images = generator(source_images, training=False)

            discriminator_predictions = discriminator(
                [
                    tf.concat([source_images, source_images], axis=0),
                    tf.concat([target_images, generated_images], axis=0)
                ],
                training=True
            )

            return base_loss_op(labels, discriminator_predictions)

        return loss_op

    def train_step(self, data):
        """
        Manual train step
        """

        source_images, target_images = data

        self.discriminator.trainable = True
        self.generator.trainable = False

        with tf.GradientTape() as discriminator_tape:

            discriminator_loss = self.discriminator_loss_op(
                source_images=source_images,
                target_images=target_images
            )

        self.discriminator_optimizer.minimize(
            discriminator_loss, self.discriminator.trainable_variables, tape=discriminator_tape)

        self.discriminator.trainable = False
        self.generator.trainable = True

        with tf.GradientTape() as generator_tape:

            generator_loss = self.generator_loss_op(
                source_images=source_images,
                target_images=target_images
            )

        self.generator_optimizer.minimize(
            generator_loss, self.generator.trainable_variables, tape=generator_tape)

        self.discriminator.trainable = False
        self.generator.trainable = False

        return {"generator_loss": generator_loss, "discriminator_loss": discriminator_loss}


class GeneratorVisualizationCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that visualizes generator output once every specified number of batches
    """

    def __init__(
            self, generator: tf.keras.Model,
            logger: net.utilities.ImagesLogger, data_iterator, logging_interval: int):
        """
        Constructor

        Args:
            generator (tf.keras.Model): generator model
            logger (photobridge.utilities.ImagesLogger): logger instance
            data_iterator: iterator that yields sources and targets images batches
            logging_interval (int): number of batches between logging
        """

        super().__init__()

        self.generator = generator
        self.logger = logger
        self.data_iterator = data_iterator
        self.logging_interval = logging_interval

        self.epoch_counter = 0
        self.batches_counter = 0

    def on_epoch_end(self, epoch: int, logs=None):
        """
        On epoch end callback
        """

        self.epoch_counter += 1

    def on_train_batch_end(self, batch, logs=None):
        """
        Visualize generator output once every x batches
        """

        if self.batches_counter == self.logging_interval:

            sources, targets = next(self.data_iterator)
            fake_targets = self.generator.predict(sources, verbose=False)

            for triplet_index, triplets in enumerate(zip(sources, fake_targets, targets)):

                title = (
                    f"Epoch {self.epoch_counter}, batch {batch} - "
                    f"sources / fake targets / targets - triplet number {triplet_index}"
                )

                self.logger.log_images(
                    title=title,
                    images=net.processing.ImageProcessor.denormalize_batch(np.array(triplets))
                )

            # Reset batch counter
            self.batches_counter = 0

        else:

            self.batches_counter += 1


class ModelCheckpoint(tf.keras.callbacks.Callback):
    """
    Callback for periodically saving model provided in constructor - so a bit different from keras' ModelCheckpoint
    """

    def __init__(self, target_model: tf.keras.Model, checkpoint_path: str, saving_interval_in_steps: int):
        """
        Constructor

        Args:
            model (tf.keras.Model): model to save
            checkpoint_path (str): path to save model at
            saving_interval_in_steps (int): specifies how often model should be saved
        """

        super().__init__()

        self.target_model = target_model
        self.checkpoint_path = checkpoint_path
        self.saving_interval_in_steps = saving_interval_in_steps

        self.steps_counter = 0

    def on_train_batch_end(self, batch, logs=None):
        """
        On train batch end callback, saves model if specified number of steps has passed since last save
        """

        if self.steps_counter == self.saving_interval_in_steps:

            shutil.rmtree(self.checkpoint_path, ignore_errors=True)
            self.target_model.save(self.checkpoint_path, save_format="h5")

            self.steps_counter = 0

        else:

            self.steps_counter += 1


class VisualizationArchivesBuilderCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that every few batches runs predictions on source images,
    then saves sources, fake targets and targets into an uncompressed tar archive.
    Once archive exceeds specified size, it's closed and rotated.
    Only max_archives_count archives are kept, and archives older than that are deleted.
    """

    def __init__(
            self, generator: tf.keras.Model, data_iterator,
            output_directory: str, file_name: str, logging_interval: int,
            max_archive_size_in_bytes: int, max_archives_count: int):
        """
        Constructor.
        Note - constructor will delete any old archives matching pattern "output_directory/filename*.tar"

        Args:
            generator (tf.keras.Model): generator model
            data_iterator: iterator that yields sources and targets images batches
            output_directory (str): directory where archives will be saved
            file_name (str): name of archive file, without extension. Extension will be .tar, and rotated archives
            will have numbers appended to them (e.g. archive.1.tar, archive.2.tar, etc.)
            logging_interval (int): number of batches between logging
            max_archive_size_in_bytes (int): maximum size of archive in bytes,
            once archive reaches this size, it's closed and rotated
            max_archives_count (int): maximum number of archives to keep,
            once this number is reached, oldest archive is deleted
        """

        super().__init__()

        self.generator = generator
        self.output_directory = output_directory
        self.file_name = file_name
        self.data_iterator = data_iterator

        self.numeric_constraints_map = {
            "logging_interval": logging_interval,
            "max_archive_size_in_bytes": max_archive_size_in_bytes,
            "max_archives_count": max_archives_count
        }

        self.counters_map = {
            "epoch": 0,
            "batch": 0
        }

        self.tar_data = {
            "base_archive_path": os.path.join(self.output_directory, f"{self.file_name}.tar"),
            "tar_files_maps": []
        }

        # Delete any old archives
        for file_path in glob.glob(f"{self.output_directory}/{self.file_name}*.tar"):
            os.remove(file_path)

    def _rotate_archives(self):
        """
        Rotate archives
        """

        # Get list of archive files
        sorted_archives_files_paths = sorted(glob.glob(os.path.join(self.output_directory, f"{self.file_name}.*.tar")))

        # We only want to keep max_archives_count, so that means that won't be
        # backing up oldest one - it will instead be overwritten by next oldest archive
        archive_files_paths_to_keep = sorted_archives_files_paths[
            :self.numeric_constraints_map["max_archives_count"] - 1]

        # Go over archives to keep after rotation from oldest to youngest
        for rotated_archive_path in reversed(archive_files_paths_to_keep):

            # Get index of archive
            index = int(os.path.basename(rotated_archive_path).split(".")[-2])

            new_archive_path = os.path.join(self.output_directory, f"{self.file_name}.{index + 1}.tar")

            # Move archive to next index
            os.rename(rotated_archive_path, new_archive_path)

        # Move latest archive to index 1
        os.rename(self.tar_data["base_archive_path"], os.path.join(self.output_directory, f"{self.file_name}.1.tar"))

    def on_epoch_end(self, epoch: int, logs=None):
        """
        On epoch end callback
        """

        self.counters_map["epoch"] += 1

    def on_train_batch_end(self, batch, logs=None):
        """
        Visualize generator output once every x batches
        """

        if self.counters_map["batch"] == self.numeric_constraints_map["logging_interval"]:

            should_rotate_archives = \
                os.path.exists(self.tar_data["base_archive_path"]) and \
                os.path.getsize(
                    self.tar_data["base_archive_path"]) > self.numeric_constraints_map["max_archive_size_in_bytes"]

            # If archive is too big, rotate archive files and clear tar files maps
            if should_rotate_archives:

                self._rotate_archives()
                self.tar_data["tar_files_maps"].clear()

            sources, targets = next(self.data_iterator)
            fake_targets = self.generator.predict(sources, verbose=False)

            # Compute tar files maps for all triplets
            for triplet_index, triplet in enumerate(zip(sources, fake_targets, targets)):

                normalized_triplet = net.processing.ImageProcessor.denormalize_batch(np.array(triplet))

                self.tar_data["tar_files_maps"].extend([
                    net.processing.get_image_tar_map(
                        image=image,
                        name=f"epoch_{self.counters_map['epoch']}_batch_{batch}_index_{triplet_index}_{name}.jpg"
                    ) for image, name in zip(normalized_triplet, ["a_source", "b_fake_target", "c_target"])
                ])

            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmp_dir:

                temporary_archive_path = os.path.join(tmp_dir, "archive.tar")

                # Create tarfile object in temporary directory
                with tarfile.open(name=temporary_archive_path, mode="x") as tar:

                    # Add all tar files maps to tar file
                    for tar_file_map in self.tar_data["tar_files_maps"]:

                        tar_file_map["bytes"].seek(0)
                        tar.addfile(tarinfo=tar_file_map["tar_info"], fileobj=tar_file_map["bytes"])

                # Move temporary file to target path
                os.rename(temporary_archive_path, self.tar_data["base_archive_path"])

            # Reset batch counter
            self.counters_map["batch"] = 0

        else:

            self.counters_map["batch"] += 1
