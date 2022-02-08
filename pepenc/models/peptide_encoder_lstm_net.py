""" An LSTM for embedding amino acid sequences
"""
import logging
logger = logging.getLogger(__name__)

import torch
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

    def forward(self, seqs, seq_lengths):
        """ Perform a forward pass to get the LSTM embedding for `seqs` """

        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat a new
        # batch as a continuation of a sequence
        # see: https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e

        #TODO: this is not required. See description of (h_0, c_0) in the LSTM documentation:
        # https://pytorch.org/docs/1.10/generated/torch.nn.LSTM.html#torch.nn.LSTM
        #self.hidden_init = self._init_hidden()

        ###
        # 1. embed the amino acids
        # 
        # Dim transformation: (batch_size, max_seq_len, 1) -> (batch_size, max_seq_len, embedding_dim)
        ###
        embeds = self.embeddings(seqs)
        
        ###
        # 2. run the embeddings through the lstm
        #
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)
        ###

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        embeds_packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=True, enforce_sorted=False)

        # we could also pass in initialization values for the hidden states here if we wanted. See the comment above
        lstm_embeds, (h_n, c_n) = self.lstm(embeds_packed)#, self.hidden_init)

        # unpack the LSTM embeddings
        lstm_unpacked, lstm_unpacked_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_embeds, batch_first=True)

        ###
        # 3. extract the output features for the last input (i.e., the index of the length) for each peptide
        ###

        # base-0        
        lstm_output_index = lstm_unpacked_lengths-1

        # by design, the indices are always on the CPU, so move them as needed
        # see: https://github.com/pytorch/pytorch/issues/7466
        lstm_output_index = lstm_output_index.to(device=lstm_unpacked.get_device())

        # additionally, we need to index each row in the output for slicing.
        # this can likely be made much more efficient (e.g., creating once and reusing)
        lstm_item_index = torch.arange(lstm_unpacked.shape[0], device=lstm_unpacked.get_device())

        # and then select the output features of the last relevant index for each sequence
        lstm_output = lstm_unpacked[lstm_item_index, lstm_output_index]
        
        return lstm_output