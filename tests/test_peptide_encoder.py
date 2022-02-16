""" A high-level set of tests for checking the peptide encoder package
"""

import pytest

import argparse
import logging
logger = logging.getLogger(__name__)

###
# If using the pyllars package, these lines can be used so that the logger
# actually outputs logging statements across all modules.
###
import pyllars.logging_utils as logging_utils
logging_utils.set_logging_values(logging_level='DEBUG')

import pepenc
import pepenc.data.data_utils as data_utils
import pepenc.models

from typing import Mapping

###
# Fixtures and other "Arrange" phase helpers
###
def get_version() -> str:
    return pepenc.__version__

def get_config() -> Mapping:
    config = data_utils.load_sample_config()
    return config

@pytest.fixture
def version() -> str:
    return get_version()

@pytest.fixture
def config() -> Mapping:
    return get_config()


###
# The actual tests
###
def test_peptide_encoder_lstm_model(config:Mapping) -> None:
    """ Ensure the training loop for the LSTM model behaves as expected """
    model = pepenc.models.PeptideEncoderLSTM(config)
    logs = model.step()

    assert len(logs) > 0

def test_version(version:str) -> None:
    """ Ensure we have the correct version

    N.B. Since we have a fixture called `version`, that will be passed as
    the "`version`" parameter when running through pytest.

    Parameters
    ----------
    version : str
        A string representation of the version of `pepenc`

    Returns
    -------
    None : None
        We assert that the version matches our expected version
    """

    # we do not really have an "Act" phase in this simple test

    # "Assert" that we have the expected behavior
    expected_version = '0.2.2'
    assert (expected_version == version)
    
def run_all():
    """ Run all of the tests

    This function is useful in case we want to run our tests outside of the pytest framework.
    """
    # since we are not running through pytest, we have to grab the inputs to the tests
    version = get_version()
    test_version(version)

    config = get_config()
    test_peptide_encoder_lstm_model(config)

if __name__ == '__main__':
    run_all()