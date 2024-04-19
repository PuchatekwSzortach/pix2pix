"""
Tests for the net.ml module
"""

import numpy as np
import pytest
import tensorflow as tf

import net.ml


class DataLoader:
    """
    Simple data loader for testing
    """

    def __init__(self, sources, targets):
        """
        Constructor
        """

        self.sources = sources
        self.targets = targets

    def __iter__(self):
        """
        Iterator for DataLoader class

        Returns:
            DataLoader: DataLoader instance
        """

        while True:
            yield self.sources, self.targets


class TestPix2Pix:
    """
    Tests for pix2pix model
    """

    @pytest.fixture(scope="session")
    def input_data(self) -> np.ndarray:
        """
        Input data for pix2pix model

        Returns:
            np.ndarray: 4D numpy array
        """

        return np.zeros((2, 256, 256, 3), dtype=np.float32)

    @pytest.fixture(scope="session")
    def pix2pix(self, input_data):
        """
        Fixture for pix2pix model
        """

        model = net.ml.Pix2PixModel(
            discriminator_patch_shape=(1, input_data.shape[1] // 8, input_data.shape[2] // 8),
            batch_size=input_data.shape[0],
            learning_rate=0.0002
        )

        model.compile()

        return model

    def test_generator_predictions(self, pix2pix, input_data):
        """
        Test generator predictions
        """

        output = pix2pix.generator.predict(input_data, verbose=False)

        # Check output has same shape as valid input
        assert output.shape == input_data.shape

    def test_discriminator_predictions(self, pix2pix, input_data):
        """
        Test discriminator predictions
        """

        output = pix2pix.discriminator.predict([input_data, input_data], verbose=False)

        expected_shape = input_data.shape[0], input_data.shape[1] // 8, input_data.shape[2] // 8, 1

        assert output.shape == expected_shape

    def test_training_model(self, pix2pix, input_data):
        """
        Test training model
        """

        training_dataset = tf.data.Dataset.from_generator(
            generator=lambda: iter(DataLoader(input_data, input_data)),
            output_types=(
                tf.float32,
                tf.float32
            ),
            output_shapes=(
                tf.TensorShape([None, None, None, 3]),
                tf.TensorShape([None, None, None, 3]),
            )
        )

        pix2pix.fit(
            x=training_dataset,
            steps_per_epoch=2,
            epochs=2,
            verbose=False
        )
