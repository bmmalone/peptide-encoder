""" This script uses [Ray tune](https://docs.ray.io/en/latest/index.html) to train peptide encoding models and optimize
hyperparameters for the models. The hyperparameter search space, as well as searching and scheduling algorithms, are
currently hard coded.
"""

import logging
import pyllars.logging_utils as logging_utils
logger = logging.getLogger(__name__)

from typing import Mapping

import argparse
import pyllars.utils

import pepenc.models

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

#TODO: this is a hack to get Ray to work on WSL with GPUs.
# for related discussion, see: https://github.com/ray-project/ray/issues/9166#issuecomment-653430294
import ray._private.resource_spec
ray._private.resource_spec._autodetect_num_gpus = lambda :  1

_DEFAULT_NAME = "my-pepenc-tune-exp"
_DEFAULT_MAX_TRAINING_ITERATIONS = 10
_DEFAULT_CHECKPOINT_FREQUENCY = 5
_DEFAULT_NUM_HYPERPARAMETER_CONFIGURATIONS = 20
_DEFAULT_OUT_DIR = "/tmp"

_VALIDATION_METRIC = "validation_median_absolute_error"
_VALIDATION_MODE = "min"

# see documentation for details: https://docs.ray.io/en/latest/tune/api_docs/execution.html
_CHECKPOINT_SCORE_ATTR = "min-validation_median_absolute_error"

def _build_search_space(config:Mapping) -> Mapping:
    """ Build the hyperparameter search space for Ray

    For more details, see: https://docs.ray.io/en/latest/tune/api_docs/search_space.html
    """
    search_space = config.copy()
    search_space['embedding_dim'] = tune.qlograndint(2*3, 2*8, 4)
    search_space['hidden_dim'] = tune.qlograndint(2*3, 2*8, 4)
    search_space['lstm_layers'] = tune.qrandint(2, 5)
    search_space['lstm_dropout'] = tune.quniform(0.2, 0.8, 0.2)
    return search_space

def optimize_hyperparameters(args:argparse.Namespace, search_space:Mapping) -> ray.tune.ExperimentAnalysis:
    """ Use Ray to optimize the hyperparameters for the pepenc embedding model
    """

    # BayesOpt does not work with integer hyperparameters
    #search_alg = BayesOptSearch(metric=_VALIDATION_METRIC, mode=_VALIDATION_MODE)
    search_alg = HyperOptSearch(metric=_VALIDATION_METRIC, mode=_VALIDATION_MODE)
    scheduler = ASHAScheduler(metric=_VALIDATION_METRIC, mode=_VALIDATION_MODE)

    analysis = tune.run(
        pepenc.models.PeptideEncoderLSTM,
        stop={"training_iteration": args.max_training_iterations},
        config=search_space,
        checkpoint_freq=args.checkpoint_frequency,
        num_samples=args.num_hyperparameter_configurations,
        resources_per_trial={"gpu": 0.3, "cpu": 1},
        scheduler=scheduler,
        search_alg=search_alg,
        
        name=args.name,

        # a directory where results are stored before being
        # sync'd to head node/cloud storage
        local_dir=args.out_dir,
        
        # sync our checkpoints via rsync
        # you don't have to pass an empty sync config - but we
        # do it here for clarity and comparison
        #sync_config=sync_config,

        # we'll keep the best five checkpoints at all times
        checkpoint_score_attr=_CHECKPOINT_SCORE_ATTR,
        keep_checkpoints_num=5,

        # a very useful trick! this will resume from the last run specified by
        # sync_config (if one exists), otherwise it will start a new tuning run
        #resume="AUTO",
        resume=False
    )

    return analysis

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('config', help="The path to the yaml configuration file.")
    parser.add_argument('--name', help="A name for the experiment", default=_DEFAULT_NAME)
    parser.add_argument('--max-training-iterations', help="The maximum number of training iterations for a single "
        "hyperparameter configuration", type=int, default=_DEFAULT_MAX_TRAINING_ITERATIONS)
    parser.add_argument('--checkpoint-frequency', help="The frequency for saving checkpoints", type=int,
        default=_DEFAULT_CHECKPOINT_FREQUENCY)
    parser.add_argument('--num-hyperparameter-configurations', help="The number of unique hyperparameter "
        "configurations to evaluate", type=int, default=_DEFAULT_NUM_HYPERPARAMETER_CONFIGURATIONS)
    parser.add_argument('--out-dir', help="The base output directory", default=_DEFAULT_OUT_DIR)

    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)
    return args

def main():
    args = parse_arguments()
    config = pyllars.utils.load_config(args.config)

    search_space = _build_search_space(config)
    optimize_hyperparameters(args, search_space)

if __name__ == '__main__':
    main()
