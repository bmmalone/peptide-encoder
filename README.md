# Peptide Encoder

An encoder for peptides (short amino acid sequences) based on BLOSUM similarity.

In particular, this package includes a model for learning peptide embeddings such that the embedding of two peptides in
the vector space is proportional to their BLOSUM62 similarity.

### Installation

This project is written in `python3` and can be installed with `pip`. It is available on PyPI.


```
pip3 install --find-links https://download.pytorch.org/whl/cu113/torch_stable.html peptide-encoder
```

Alternatively, the package can be installed from source.

```
git clone https://github.com/bmmalone/peptide-encoder.git
cd piptide-encoder
pip3 install -r requirements.txt .
```

(The "period" at the end is required.)

**Prerequisites**: This project relies on quite a few prerequisites, such as pytorch, ray, cudnn, and others. Both the
`requirements.txt` and `setup.py` files aim to install these dependencies correctly; nevertheless, it may still be
preferable to install these dependencies before installing this package.

In particular, the `find-links` argument to pip may need to be adjusted depending on the available version of CUDA.

### Usage

After installation, models can be trained using a command similar to the following:

```
train-pepenc-models /prj/peptide-encoder/conf/base/config.yaml --num-hyperparameter-configurations 500 --max-training-iterations 30 --name my-pepenc-tune-exp --out-dir /tmp
```

The `--help` flag can be used to see a description of all arguments to the script.

For adjusting the hyperparameter search space, algorithms, or schedulers, the `pepenc/models/train_pepenc_models.py`
script can be adjusted. If the package was not installed in `pip` "editable" mode, then make sure to re-run `pip install`
so that the changes take effect for the next run of ray.

### Documentation

Unfortunately, there is no sphinx, etc., documentation at this time. The file `conf/base/config.yaml` shows examples of
all hyperparameters, data files, etc., for training models.

#### Data format

The models in this project require an input csv file that has one row which is a header and remaining rows which are
the peptides for the various datasets. The column in the csv file with the peptide sequences must be named `sequence`.
(This can be adjusted if calling the `pepenc` library from python code.)

### Tensorboard visualization

The `<out_dir>/<name>` directory (based on the arguments to `train-pepenc-models`) will contain output suitable for
visualization with Tensorboard. The following command uses Docker to expose the results on port 6007.

```
docker run --rm --mount type=bind,source=/tmp/my-pepenc-tune-exp,target=/tensorboard --publish 6007:6006 nvcr.io/nvidia/tensorflow:21.12-tf2-py3 tensorboard --logdir /tensorboard
```

The tensorflow image can be updated as necessary.

**N.B.** The `source` of the bind mount must be the `<out_dir>/<name>` directory (based on the arguments to `train-pepenc-models`).

### Training the model

The model consistently experiences vanishing (or, improbably, exploding) gradient issues when using a single LSTM layer.
It is not clear why this happens, and it is currently suggested to avoid allowing `lstm_layers == 1` in the
hyperparameter search space (or directly setting it that way in the config).

### Testing the code

The project uses `pytest` for unit testing. The testing prerequisites (though not other dependencies, as described
above) can be installed as follows.

```
pip3 install .[test]
```

After installing `pytest` and other testing dependencies, tests can be performed as follows.

```
cd /path/to/peptide-encoder
pytest .
```