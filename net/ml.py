"""
Module with machine training logic
"""

import typing

import tensorflow as tf


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
        Get pixp2pix generator model
        """

        input_op = tf.keras.layers.Input(shape=(None, None, 3))

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="same")(input_op)

        def downscale_block(input_op, filters: int, use_activation: bool, use_normalization: bool):
            """
            Downscale block
            """

            x = input_op

            if use_activation is True:
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=4, strides=(2, 2), padding="same")(input_op)

            if use_normalization is True:
                x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)

            return x

        def upscale_block(input_op, skip_input: bool, filters: int, use_dropout: bool):
            """
            Upscale block
            """

            x = tf.keras.layers.ReLU()(input_op)

            x = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=4,
                strides=(2, 2),
                padding="same")(x)

            x = tf.keras.layers.Concatenate()([x, skip_input])

            x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)

            if use_dropout is True:
                x = tf.keras.layers.Dropout(rate=0.5)(x)

            return x

        downscale_ops = {
            # Output is 2x smaller
            1: downscale_block(input_op=input_op, filters=64, use_activation=False, use_normalization=False)
        }

        filters_counts = [128, 256, 512, 512, 512, 512]

        for index, filters_count in enumerate(filters_counts):

            downscale_ops[index + 2] = downscale_block(
                input_op=downscale_ops[index + 1],
                filters=filters_count,
                use_activation=True,
                use_normalization=True
            )

        innermost_layer = downscale_block(
            input_op=downscale_ops[7],
            filters=512,
            use_activation=True,
            use_normalization=False
        )

        upscale_ops = {
            # Output is 64x smaller
            7: upscale_block(
                input_op=innermost_layer, skip_input=downscale_ops[7], filters=512, use_dropout=True)
        }

        # # Output is 64x smaller
        upscale_ops[6] = upscale_block(
            input_op=upscale_ops[7], skip_input=downscale_ops[6], filters=512, use_dropout=True)

        # Output is 32x smaller
        upscale_ops[5] = upscale_block(
            input_op=upscale_ops[6], skip_input=downscale_ops[5], filters=512, use_dropout=True)

        for index, filters_count in enumerate([512, 256, 128, 64]):

            upscale_ops[4 - index] = upscale_block(
                input_op=upscale_ops[5 - index],
                skip_input=downscale_ops[4 - index],
                filters=filters_count,
                use_dropout=True
            )

        x = tf.keras.layers.ReLU()(upscale_ops[1])

        # Output is same as input
        output_op = tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            activation="tanh")(x)

        model = tf.keras.models.Model(input_op, output_op)

        # Compile model so we can save it without keras throwing warnings
        model.compile()

        return model

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
