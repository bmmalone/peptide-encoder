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

@pytest.fixture
def config() -> Mapping:
    return get_config()


def test_load_peptide_encoder_lstm_model() -> None:
    """ Load a model """

    # see: https://github.com/ray-project/ray/issues/21212
    from ray.rllib import register_trainable
    register_trainable("PeptideEncoderLSTM", PeptideEncoderLSTM)

    checkpoint_folder = pathlib.Path("/tmp/my-quick-pepenc-tune-exp/")

    # load the existing analysis
    analysis = ExperimentAnalysis(str(checkpoint_folder))
    
    # get the best result
    best_trial = analysis.get_best_trial(metric="validation_median_absolute_error", mode="min", scope="all") 
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="validation_median_absolute_error", mode='min')

    model = PeptideEncoderLSTM.load(best_checkpoint)
    val_results = model._val_step()

    assert val_results['validation_median_absolute_error'] < 0.1

def test_load_peptide_encoder_lstm_model_class_method() -> None:
    checkpoint_folder = pathlib.Path("/tmp/my-quick-pepenc-tune-exp/")
    model = PeptideEncoderLSTM.load_from_ray_results(checkpoint_folder)
    val_results = model._val_step()
    assert val_results['validation_median_absolute_error'] < 0.1


    
def run_all():
    """ Run all of the tests

    This function is useful in case we want to run our tests outside of the pytest framework.
    """
    # since we are not running through pytest, we have to grab the inputs to the tests
    config = get_config()

    test_load_peptide_encoder_lstm_model_class_method()
    test_load_peptide_encoder_lstm_model()

if __name__ == '__main__':
    run_all()