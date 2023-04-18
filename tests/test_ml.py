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
    def pix2pix(self):
        """
        Fixture for pix2pix model
        """

        return net.ml.Pix2PixModel()

    def test_generator_predictions(self, pix2pix):
        """
        Test generator predictions
        """

        data = np.zeros((2, 256, 256, 3), dtype=np.float32)

        output = pix2pix.generator.predict(data, verbose=False)

        # Check output has same shape as valid input
        assert output.shape == data.shape
