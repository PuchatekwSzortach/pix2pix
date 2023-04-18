"""
Tests for the net.ml module
"""

import numpy as np
import pytest

import net.ml


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

        return net.ml.Pix2PixModel(
            discriminator_patch_shape=(1, input_data.shape[1], input_data.shape[2]),
            batch_size=input_data.shape[0],
        )

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
