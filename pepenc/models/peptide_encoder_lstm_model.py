""" This module contains the wrapper class for the PeptideEncoderLSTM network. It handles issues like loading data,
training, prediction, and disk io.
"""
import logging
logger = logging.getLogger(__name__)

import joblib
import numpy as np
import pathlib
import sklearn.metrics
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim


import ray.tune
from ray.tune import ExperimentAnalysis

import toolz.dicttoolz as dicttoolz

import pyllars.torch.torch_utils as torch_utils
import pyllars.utils

import pepenc.pepenc_utils as pepenc_utils
from pepenc.data.peptide_encoder_training_dataset import (
    PeptideEncoderTrainingDataset, PeptideEncoderTrainingDatasetItem
)
from pepenc.models.peptide_encoder_lstm_net import PeptideEncoderLSTMNetwork

from typing import Dict, Mapping, NamedTuple

_DEFAULT_BATCH_SIZE = 64
_DEFAULT_ADAM_BETA1 = 0.9
_DEFAULT_ADAM_BETA2 = 0.999
_DEFAULT_ADAM_LR = 0.01
_DEFAULT_LR_PATIENCE = 3
_DEFAULT_WEIGHT_DECAY = 0

_EPOCH_SIZE = 8192
_TEST_SIZE = 4096

_DEFAULT_NAME = "PeptideEncoderLSTM"

class PeptideEncoderLSTMPredictionResults(NamedTuple):
    all_sequence_px: np.ndarray
    all_sequence_py: np.ndarray
    all_encoded_px: np.ndarray
    all_encoded_py: np.ndarray
    all_embedded_px: np.ndarray
    all_embedded_py: np.ndarray
    all_similarity: np.ndarray
    all_embedded_similarity: np.ndarray

class _PeptideEncoderLSTMDeviceDataStructures(NamedTuple):
    all_embedded_px: torch.FloatTensor
    all_embedded_py: torch.FloatTensor
    all_embedded_similarity: torch.FloatTensor
    

class PeptideEncoderLSTM(ray.tune.Trainable):
    """ The PeptideEncoderLSTM training model

    Parameters
    ----------
    config : typing.Mapping
        The configuration options. Please see the
        aa_encoder.aa_encoder_net.AAEncoderNetwork documentation for further
        configuration options.

        * `mode`: valid options: "train", "pred"
        * `batch_size`: the training batch size
        * `device_name`: the name of the device for torch. Optional.
        * `name`: a name for this model. This is mostly used for logging
        * `aa_encoding_map`: the path to a joblib'd dictionary containing the index encoding for each amino acid

        N.B. Configuration options are currently not validated.
    """
    def setup(self, config:Mapping) -> None:
        """ Build the model according to `config` """
        self.config = config
        self.name = config.get('name', _DEFAULT_NAME)
        self.batch_size = config.get('batch_size', _DEFAULT_BATCH_SIZE)
        self.device_name = config.get('device_name')

        self.device = torch_utils.get_device(self.device_name)

        self._load_all_datasets()
        self._prepare_network()
        self._prepare_loss()
        self._prepare_optimizer()

    def log(self, msg:str, level:int=logging.INFO):    
        """ Log `msg` using `level` using the module-level logger """    
        msg = "[{}] {}".format(self.name, msg)
        logger.log(level, msg)


    def get_network_type(self):
        """ Retrieve the type of the network underlying the model """
        return PeptideEncoderLSTMNetwork

    def get_optimizer_type(self):
        """ Retrieve the type of optimizer used for the model """
        return optim.Adam
        
    def _load_all_datasets(self):
        """ Load all of the datasets specified in the configuration file """
        aa_encoding_map = self.config.get('aa_encoding_map')
        if aa_encoding_map is not None:
            self.aa_encoding_map = joblib.load(aa_encoding_map)
        else:
            msg = "aa_encoding_map was not given in the configuration"
            self.log(msg, logging.ERROR)
            raise KeyError(msg)

        #TODO: maybe there is a better way to do this
        self.config['vocabulary_size'] = len(self.aa_encoding_map)

        self.training_set = PeptideEncoderTrainingDataset.load(
            self.config.get('training_set'), self.aa_encoding_map, is_validation=False, name="TrainingDataset"
        )
        self.validation_set = PeptideEncoderTrainingDataset.load(
            self.config.get('validation_set'), self.aa_encoding_map, is_validation=True, name="ValidationDataset"
        )
        self.test_set = PeptideEncoderTrainingDataset.load(
            self.config.get('test_set'), self.aa_encoding_map, is_validation=True, name="TestDataset"
        )

    def _prepare_network(self):
        """ Create the underlying neural network and set it to the `device` """
        net = PeptideEncoderLSTMNetwork(self.config)
        self.net = torch_utils.send_network_to_device(net, self.device)
        
        logger.debug(self.net)

    def _prepare_loss(self):
        """ Create the loss function """
        self.loss_function = nn.MSELoss()

    def _prepare_optimizer(self):
        """ Create the optimizer """
        lr = self.config.get('adam_lr', _DEFAULT_ADAM_LR)
        weight_decay = self.config.get('weight_decay', _DEFAULT_WEIGHT_DECAY)
        adam_beta1 = self.config.get("adam_beta1", _DEFAULT_ADAM_BETA1)
        adam_beta2 = self.config.get("adam_beta2", _DEFAULT_ADAM_BETA2)
        adam_betas = (adam_beta1, adam_beta2)

        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=adam_betas
        )

        lr_patience = self.config.get("lr_patience", _DEFAULT_LR_PATIENCE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=lr_patience
        )

    def step(self):
        train_logs = self._train_step()
        val_logs = self._val_step()
        logs = dicttoolz.merge(train_logs, val_logs)
        return logs

    def _train_step(self):
        """ Perform one round of training """

        self.net.train()
        train_loader = torch.utils.data.DataLoader(
            self.training_set, batch_size=self.config.get('batch_size'), shuffle=True
        )

        train_loss = 0.0
        one = torch.as_tensor(1, dtype=torch.float32)
        one = one.to(self.device)

        for batch_idx, data in enumerate(train_loader):
            
            # We set this just for the example to run quickly.
            if batch_idx * len(data) > _EPOCH_SIZE:
                return
               
            # send this instance to the device
            data = torch_utils.send_data_to_device(*data, device=self.device)
            peptide_x, peptide_y, px, py, similarity = data
            
            # zero out the parameter gradients
            self.optimizer.zero_grad()
            
            x_lengths = self.training_set.get_trimmed_peptide_lengths(peptide_x)
            embedded_x = self.net(px, x_lengths)


            y_lengths = self.training_set.get_trimmed_peptide_lengths(peptide_y)
            embedded_y = self.net(py, y_lengths)

            embedded_distance = pepenc_utils.calculate_matched_minkowski_distances(embedded_x, embedded_y)
            embedded_similarity = one - embedded_distance

            # calculate the loss
            loss = self.loss_function(embedded_similarity, similarity)

            # and back prop
            loss.backward()
            self.optimizer.step()

            # pull out the loss
            loss = loss.item()
            train_loss += loss

        # find the average loss for each batch
        train_loss_mean = train_loss / float(batch_idx)

        # summarize performance on the training set for this step
        train_summary = {
            'loss': train_loss_mean
        }

        return train_summary

    def _pred(self,
            dataset:PeptideEncoderTrainingDataset,
            include_embeddings:bool=False,
            progress_bar:bool=False) -> PeptideEncoderLSTMPredictionResults:

        self.net.eval() # make sure the network knows we are making predictions

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.config.get('batch_size'), shuffle=False
        )

        it = enumerate(data_loader)
        if progress_bar:
            it = tqdm.tqdm(it)

        # keep track of all predictions
        all_sequence_px = []
        all_sequence_py = []
        all_encoded_px = []
        all_encoded_py = []
        all_similarity = []
        all_embedded_similarity = []
        all_embedded_px = []
        all_embedded_py = []

        one = torch.as_tensor(1, dtype=torch.float32)

        for i, data in it:
            data_to_device = torch_utils.send_data_to_device(
                *data, device=self.device
            )
            data_to_device = PeptideEncoderTrainingDatasetItem(*data_to_device)

            x_lengths = dataset.get_trimmed_peptide_lengths(data_to_device.aa_sequence_xs)
            y_lengths = dataset.get_trimmed_peptide_lengths(data_to_device.aa_sequence_ys)

            embedded_x = self.net(data_to_device.encoded_xs, x_lengths)
            embedded_y = self.net(data_to_device.encoded_ys, y_lengths)

            embedded_distance = pepenc_utils.calculate_matched_minkowski_distances(embedded_x, embedded_y)
            embedded_similarity = one - embedded_distance

            device_data = _PeptideEncoderLSTMDeviceDataStructures(
                all_embedded_px=embedded_x,
                all_embedded_py=embedded_y,
                all_embedded_similarity=embedded_similarity
            )

            data_from_device = torch_utils.retrieve_data_from_device(*device_data)
            data_from_device = _PeptideEncoderLSTMDeviceDataStructures(*data_from_device)

            all_sequence_px.append(data.aa_sequence_xs)
            all_sequence_py.append(data.aa_sequence_ys)
            all_encoded_px.append(data.encoded_xs)
            all_encoded_py.append(data.encoded_ys)
            all_similarity.append(data.similarities)
            all_embedded_similarity.append(
                data_from_device.all_embedded_similarity
            )

            if include_embeddings:
                all_embedded_px.append(data_from_device.all_embedded_px)
                all_embedded_py.append(data_from_device.all_embedded_py)

        # we need to convert the input data back from torch to numpy...
        ret = PeptideEncoderLSTMPredictionResults(
            np.concatenate(all_sequence_px),
            np.concatenate(all_sequence_py),
            torch_utils.tensor_list_to_numpy(torch.cat(all_encoded_px)),
            torch_utils.tensor_list_to_numpy(torch.cat(all_encoded_py)),
            np.concatenate(all_embedded_px),
            np.concatenate(all_embedded_py),
            torch_utils.tensor_list_to_numpy(torch.cat(all_similarity)),
            np.concatenate(all_embedded_similarity)
        )

        return ret

    def _val_step(self, prefix:str="validation_", progress_bar:bool=False) -> Dict[str, float]:
        """ Make predictions on `dataset` and summarize the performance

        Parameters
        ----------
        prefix : str
            A prefix can be added to each key in `metrics`

        progress_bar : bool
            Whether to show a progress bar while making the predictions

        Returns
        -------
        metrics : typing.Dict
            A dictionary with the following metrics. All keys will have `prefix` as a prefix.

            * mean_squared_error
            * median_absolute_error
        """

        # make the predictions
        preds = self._pred(self.validation_set, progress_bar=progress_bar, include_embeddings=True)
        
        # calculate all metrics
        mse = sklearn.metrics.mean_squared_error(
            y_true=preds.all_similarity,
            y_pred=preds.all_embedded_similarity
        )

        mae = sklearn.metrics.median_absolute_error(
            y_true=preds.all_similarity,
            y_pred=preds.all_embedded_similarity
        )

        # and build up the result summary
        ret = {
            '{}mean_squared_error'.format(prefix): mse,
            '{}median_absolute_error'.format(prefix): mae,
        }

        return ret

    def save_checkpoint(self, checkpoint_dir):
        """ Save the model and optimizer to `checkpoint_dir` """
        path = torch_utils.save_model(self, checkpoint_dir)
        return path

    def load_checkpoint(self, checkpoint_dir):
        torch_utils.restore_model(self, checkpoint_dir)

    @classmethod
    def load(clazz, ray_checkpoint_file:str) -> "PeptideEncoderLSTM":
        """ Load a model associated with the given checkout file

        Parameters
        ----------
        ray_checkpoint_file : str
            The full path to the Ray checkpoint file; this should be the actual `checkpoint.pt` file, not the directory.

        Returns
        -------
        model : pepenc.models.PeptideEncoderLSTM
            The model associated with the checkpoint
        """
        p = pathlib.Path(ray_checkpoint_file)

        config = p.parent / "params.json"
        config = pyllars.utils.load_config(config)

        model = clazz(config)
        model.load_checkpoint(p.parent)
        return model

    @classmethod
    def load_from_ray_results(clazz, ray_experiment_folder:str, metric:str="validation_median_absolute_error",
            mode:str="min") -> "PeptideEncoderLSTM":
        """ Load the best model from Ray results

        Parameters
        ----------
        ray_experiment_folder : str
            The path to the Ray experiment results

        metric, mode: str
            The criteria for selecting the best model

        Returns
        -------
        model : pepenc.models.PeptideEncoderLSTM
            The model associated with the best checkpoint
        """
            
        # see: https://github.com/ray-project/ray/issues/21212
        from ray.rllib import register_trainable
        register_trainable("PeptideEncoderLSTM", PeptideEncoderLSTM)

        # load the existing analysis
        analysis = ExperimentAnalysis(ray_experiment_folder)
        
        # get the best result
        best_trial = analysis.get_best_trial(metric=metric, mode=mode, scope="all") 
        # Gets best checkpoint for trial based on accuracy.
        best_checkpoint = analysis.get_best_checkpoint(best_trial, metric=metric, mode=mode)

        model = PeptideEncoderLSTM.load(best_checkpoint)

        return model