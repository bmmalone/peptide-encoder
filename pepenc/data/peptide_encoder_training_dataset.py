""" This module contains a pytorch dataset for learning peptide embeddings.

In particular, each "instance" of the dataset comprises two peptide sequences,
as well as the sNebula similarity between them. The sNebula distance reflects
the BLOSSUM similarity transformed from 0 to 1.
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np

import torch
import torch.utils.data

from lifesci.peptide_dataset import PeptideDataset

import lifesci.sequence_similarity_utils as sequence_similarity_utils
import pyllars.string_utils as string_utils

from typing import NamedTuple, Optional

class PeptideEncoderTrainingDatasetItem(NamedTuple):
    aa_sequence_xs: str
    aa_sequence_ys: str
    encoded_xs: torch.IntTensor
    encoded_ys: torch.IntTensor
    similarities: torch.FloatTensor

_DEFAULT_SEQUENCE_COLUMN = 'sequence'
_DEFAULT_SEED = 8675309
_DEFAULT_NAME = "PeptideEncoderTrainingDataset"
_DEFAULT_MAX_LEN = 25


class PeptideEncoderTrainingDataset(torch.utils.data.Dataset):
    """ Generate training samples from a list of amino acid sequences

    In particular, this class reads a list of peptides from `dataset_path`. It
    then draws pairs of peptides from the list and calculates the sNebula
    similarity score between them. Thus, each item from this dataset consists
    of two peptide sequences and the similarity score.

    In case the dataset object should be used for validation, the
    `is_validation` flag can be set to `True`. In that case, a fixed set of
    pairings will be selected for the peptides so that performance metrics are
    constant from iteration to iteration. Otherwise (i.e., for training), one
    member of each pair is randomly sampled.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset. It should be compatible with `pandas.read_csv`
        and contain a column named `sequence_column` which includes the
        sequences. Other columns are ignored.

    aa_encoding_map : pyllars.string_utils.encoding_map_type
        A mapping from each amino acid to its integer index.

        N.B. This should **not** be a one-hot representation, but, as stated,
        the integer index. Further, the padding character must be "-".

    is_validation : bool
        Whether the dataset will be used for validation (or testing)

    sequence_column : str
        The name of the column which contains the amino acid sequences

    max_len : int
        The maximum length for a peptide. Peptides longer than this will be
        truncated, and shorter peptides will be padded to this length.

    seed : int
        Seed for the random number generator. This is used to randomly select
        the second sequence in each of the instances.

    name : str
        A name for the dataset instance. This is mostly used for logging.
    """
    
    def __init__(self,
            dataset_path:str,
            aa_encoding_map:string_utils.encoding_map_type,
            is_validation:bool=False,
            sequence_column:str=_DEFAULT_SEQUENCE_COLUMN,
            max_len:int=_DEFAULT_MAX_LEN,
            seed:int=_DEFAULT_SEED,
            name:str=_DEFAULT_NAME):

        self.aa_encoding_map = aa_encoding_map
        self.is_validation = is_validation
        self.sequence_column = sequence_column
        self.max_len = max_len        
        self.seed = seed
        self.name = name
        self.rng = np.random.default_rng(self.seed)

        df_peptides = PeptideDataset.load(dataset_path, sequence_column, filters=["standard_aa_only"])
        self.aa_sequences = df_peptides[self.sequence_column].values

        self.encoded_aa_sequences = string_utils.encode_all_sequences(
            sequences=self.aa_sequences,
            encoding_map=self.aa_encoding_map,
            maxlen=self.max_len,
            pad_value='-',
            same_length=False
        )
        self.encoded_aa_sequences = self.encoded_aa_sequences.astype(int)

        if self.is_validation:
            self._matching_validation_item = np.random.permutation(len(self.aa_sequences))
        

    def log(self, msg:str, level:int=logging.INFO) -> None:
        """ Log `msg` using `level` using the module-level logger """    
        msg = "[{}] {}".format(self.name, msg)
        logger.log(level, msg)

    def __len__(self) -> int:
        return len(self.aa_sequences)

    def __getitem__(self, idx) -> PeptideEncoderTrainingDatasetItem:
        x = idx

        # and choose an appropriate matching index based on the dataset status
        if self.is_validation:
            y = self._matching_validation_item[idx]
        else:
            # select the second sequence randomly
            y = self.rng.integers(low=0, high=len(self), size=1)
            # the rng returns an array...
            y = y[0]
        
        encoded_xs = self.encoded_aa_sequences[x]
        encoded_ys = self.encoded_aa_sequences[y]

        peptide_xs = self.aa_sequences[x]
        peptide_ys = self.aa_sequences[y]
        similarities = sequence_similarity_utils.get_snebula_score(peptide_xs, peptide_ys)

        encoded_xs = torch.as_tensor(encoded_xs, dtype=torch.long)
        encoded_ys = torch.as_tensor(encoded_ys, dtype=torch.long)
        similarities = torch.as_tensor(similarities, dtype=torch.float32)
        ret = PeptideEncoderTrainingDatasetItem(
            peptide_xs, peptide_ys, encoded_xs, encoded_ys, similarities
        )

        return ret

    def get_trimmed_peptide_lengths(self, peptides) -> np.ndarray:
        """ Extract the trimmed length of the given peptides, which accounts for max_len """
        peptide_lengths = [len(p) for p in peptides]
        trimmed_peptide_lengths = np.clip(peptide_lengths, 0, self.max_len)
        return trimmed_peptide_lengths
        
    @classmethod
    def load(clazz,
            dataset_path:Optional[str],
            aa_encoding_map:string_utils.encoding_map_type,
            is_validation:bool,
            name:str) -> Optional["PeptideEncoderTrainingDataset"]:
        """ Load the dataset given by `key` in `self.config`
        
        Additionally, `name` will be used for the name of the dataset.

        Parameters
        ----------
        dataset_path : typing.Optional[str]
            The path to the dataset

        aa_encoding_map : pyllars.string_utils.encoding_map_type
            A mapping from each amino acid to its integer index.

        is_validation : bool
        Whether the dataset will be used for validation (or testing)

        name : str
            The name for the dataset, if it is in the config file. Example:
            "TrainingSet"

        Returns
        -------
        dataset : typing.Optional[AAEncoderDataset]
            If `key` is in `self.config`, then `dataset` will be the dataset
            object based on that file. Otherwise, this function returns `None`.
        """
        dataset = None

        if dataset_path is not None:
            dataset = PeptideEncoderTrainingDataset (
                dataset_path=dataset_path,
                aa_encoding_map=aa_encoding_map,
                is_validation=is_validation,
                name=name
            )

        return dataset
