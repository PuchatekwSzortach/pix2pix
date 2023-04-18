"""
Module with machine training logic
"""

import tensorflow as tf


class Pix2PixModel(tf.keras.Model):
    """
    Pix2Pix model
    """

    def __init__(self) -> None:

        super().__init__()

        self.generator = self._get_generator()

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
