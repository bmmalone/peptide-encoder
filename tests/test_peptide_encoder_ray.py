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

import pathlib
import pepenc.data.data_utils as data_utils

from ray.tune import ExperimentAnalysis

from pepenc.models.peptide_encoder_lstm_model import PeptideEncoderLSTM

from typing import Mapping

def get_config() -> Mapping:
    config = data_utils.load_sample_config()
    return config

def get_ray_experiment_folder() -> str:
    ray_experiment_folder = data_utils.get_sample_ray_experiment_path()
    return ray_experiment_folder

@pytest.fixture
def config() -> Mapping:
    return get_config()

@pytest.fixture
def ray_experiment_folder() -> str:
    return get_ray_experiment_folder()

def test_load_peptide_encoder_lstm_model_class_method(ray_experiment_folder:str) -> None:
    model = PeptideEncoderLSTM.load_from_ray_results(ray_experiment_folder)
    val_results = model._val_step()
    assert val_results['validation_median_absolute_error'] < 0.1

    
def run_all():
    """ Run all of the tests

    This function is useful in case we want to run our tests outside of the pytest framework.
    """
    # since we are not running through pytest, we have to grab the inputs to the tests
    config = get_config()
    ray_experiment_folder = get_ray_experiment_folder()

    test_load_peptide_encoder_lstm_model_class_method(ray_experiment_folder)

if __name__ == '__main__':
    run_all()