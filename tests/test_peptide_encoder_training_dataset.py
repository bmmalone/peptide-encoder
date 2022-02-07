""" Tests for validating padding and packing for the training dataset
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

import numpy as np
import torch

from lifesci.peptide_dataset import PeptideDataset
import pyllars.string_utils as string_utils

import pepenc.data.data_utils as data_utils
from pepenc.data.peptide_encoder_training_dataset import PeptideEncoderTrainingDataset

from typing import Mapping

###
# Fixtures and other "Arrange" phase helpers
###
def get_config() -> Mapping:
    config = data_utils.load_sample_config()
    return config

@pytest.fixture
def config() -> Mapping:
    return get_config()

###
# The actual tests
###
def test_encoding_peptides_with_padding(config:Mapping) -> None:
    """ Test that encoding the peptides after adding padding behaves as expected """

    dataset_path = config.get("training_set")
    sequence_column = "sequence"
    aa_encoding_map = data_utils.load_encoding_map()
    maxlen = config.get('max_sequence_length')

    # for testing, just use a small subset
    #num_peptides= 100

    # load the data
    df_peptides = PeptideDataset.load(dataset_path, sequence_column)
    #df_peptides = df_peptides.sample(n=num_peptides)
    aa_sequences = df_peptides[sequence_column].values

    # actually run the padding and encoding function
    encoded_sequences = string_utils.encode_all_sequences(
        sequences=aa_sequences,
        encoding_map=aa_encoding_map,
        maxlen=maxlen,
        pad_value='-',
        same_length=False,
        progress_bar=True
    )

    # and check that the output size matches what we expect
    expected_shape = (len(aa_sequences), maxlen)
    assert (encoded_sequences.shape == expected_shape)

def test_peptide_encoder_training_dataset_padding(config:Mapping) -> None:
    """ Ensure the training loop for the LSTM model behaves as expected """
    aa_encoding_map = data_utils.load_encoding_map()

    # create the training dataset and loader
    training_set = PeptideEncoderTrainingDataset.load(
        config.get('training_set'), aa_encoding_map, "TrainingDataset"
    )

    train_loader = torch.utils.data.DataLoader(
        training_set, batch_size=config.get('batch_size'), shuffle=True
    )

    # just check the first batch
    data = next(iter(train_loader))
    peptide_x, peptide_y, px, py, similarity = data
    x_lengths = training_set.get_trimmed_peptide_lengths(peptide_x)
            
    # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
    X_packed = torch.nn.utils.rnn.pack_padded_sequence(px, x_lengths, batch_first=True, enforce_sorted=False)

    # check that the packed data contains the number of elements we expected
    num_data_items = X_packed.data.size()[0]
    expected_num_data_items = sum(x_lengths)
    assert num_data_items == expected_num_data_items

    # then unpack to make sure we get our original data back out
    X_unpacked, X_unpacked_lengths = torch.nn.utils.rnn.pad_packed_sequence(X_packed, batch_first=True)

    # and check that the lengths match our original lengths
    assert np.array_equal(x_lengths, X_unpacked_lengths)

def run_all():
    """ Run all of the tests

    This function is useful in case we want to run our tests outside of the pytest framework.
    """
    # since we are not running through pytest, we have to grab the inputs to the tests
    config = get_config()

    test_encoding_peptides_with_padding(config)
    test_peptide_encoder_training_dataset_padding(config)

if __name__ == '__main__':
    run_all()