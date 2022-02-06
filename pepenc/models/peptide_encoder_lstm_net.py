""" An LSTM for embedding amino acid sequences
"""
import logging
logger = logging.getLogger(__name__)

import torch.nn as nn

import pyllars.validation_utils as validation_utils
from typing import Mapping

# these defaults are very low so CPU-based demos are quick
_DEFAULT_EMBEDDING_DIM:int = 4
_DEFAULT_HIDDEN_DIM:int = 4
_DEFAULT_LAYERS:int = 1
_DEFAULT_DROPOUT:float = 0.5

_DEFAULT_NAME = "PeptideEncoderLSTMNetwork"

class PeptideEncoderLSTMNetwork(nn.Module):
    """ This class implements a simple LSTM for encoding amino acid sequences.
    It takes as input a sequence of tokens (presumably, amino acids). The
    sequence tokens are assumed to already be converted to indices appropriate
    for use in an embedding layer.

    This model is very simple and does not expose very many hyperparameters.

    Parameters
    ----------
    config : typing.Mapping
        The configuration options for this network. See the "Attributes" below
        for valid options.

    name : str
        A name for the network instance. This is mostly used for logging.

    Attributes
    ----------
    embedding_dim : int
        The dimensionality for embedding the tokens

    hidden_dim : int
        The dimensionality of the hidden state in the LSTM
        
    lstm_layers : int
        The number of layers in the LSTM
        
    lstm_dropout : float

    vocabulary_size : int
        The number of unique tokens in the sequences
    """
    
    def __init__(self, config:Mapping, name:str=_DEFAULT_NAME):
        super(PeptideEncoderLSTMNetwork, self).__init__()
        self.config = config
        self.name = name

        self.embedding_dim = config.get('embedding_dim', _DEFAULT_EMBEDDING_DIM)
        self.hidden_dim = config.get('hidden_dim', _DEFAULT_HIDDEN_DIM)
        self.layers = config.get('lstm_layers', _DEFAULT_LAYERS)
        self.dropout = config.get('lstm_dropout', _DEFAULT_DROPOUT)
        self.vocabulary_size = config.get('vocabulary_size')

        self._validate_hyperparameters()

        self._build_network()

    def log(self, msg:str, level:int=logging.INFO):    
        """ Log `msg` using `level` using the module-level logger """    
        msg = "[{}] {}".format(self.name, msg)
        logger.log(level, msg)

    def _validate_hyperparameters(self):
        """ Ensure all of the options from the configuration were valid """
        validation_utils.validate_type(self.vocabulary_size, [int], "vocabular_size", self.name)

    def _build_network(self):
        """ Create the layers for the encoding and LSTM network """
        self.embeddings = nn.Embedding(self.vocabulary_size, self.embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            dropout=self.dropout
        )

    def forward(self, seq):
        """ Perform a forward pass to get the LSTM embedding for `seq` """
        # embed the words
        embeds = self.embeddings(seq)
        
        # run the embeddings through the lstm
        
        # this gives (all_hidden_states, last_state)
        all_hidden_states, last_state = self.lstm(embeds.view(len(seq), 1, -1))
        last_hidden_state, last_cell_state = last_state
        
        return last_hidden_state