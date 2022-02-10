""" This module contains helpers for automatically loading data files stored
in the aa_encode/data folder.
"""
import joblib
import os
import pathlib
import yaml

from typing import Mapping

def _get_base_data_dir() -> pathlib.Path:
    import pepenc.data # that's me!
    base_data_dir = os.path.dirname(pepenc.data.__file__)
    base_data_dir = pathlib.Path(base_data_dir)
    #base_data_dir = base_data_dir.parent
    return base_data_dir

###
# Paths to files in the `aa_encode/data` directory
###

def get_sample_config_path():
    path = _get_base_data_dir() / "conf" / "base" / "config.yaml"
    return str(path)

def get_encoding_map_path():
    path = _get_base_data_dir() / "intermediate" / "oh-aa-encoding-map.jpkl"
    return str(path)

def get_sample_peptides_path():
    path = _get_base_data_dir() / "raw" / "sample-peptides.training.csv"
    return str(path)

def get_sample_training_peptides_path():
    path = _get_base_data_dir() / "raw" / "sample-peptides.training.csv"
    return str(path)

def get_sample_test_peptides_path():
    path = _get_base_data_dir() / "raw" / "sample-peptides.test.csv"
    return str(path)

def get_sample_validation_peptides_path():
    path = _get_base_data_dir() / "raw" / "sample-peptides.validation.csv"
    return str(path)

###
# Helpers for loading the distributed data files
###
def load_encoding_map() -> Mapping:
    p = get_encoding_map_path()
    encoding_map = joblib.load(p)
    return encoding_map

def load_sample_config(update_paths:bool=True) -> Mapping:
    """ Load the config file and, optionally, update the path values"""
    path = get_sample_config_path()

    with open(path) as f:
        config = yaml.full_load(f)

    if update_paths:
        config['aa_encoding_map'] = get_encoding_map_path()
        config['training_set'] = get_sample_training_peptides_path()
        config['validation_set'] = get_sample_validation_peptides_path()
        config['test_set'] = get_sample_test_peptides_path()

    return config