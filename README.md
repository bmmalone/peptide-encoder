# Peptide Encoder

An encoder for peptides (short amino acid sequences) based on BLOSUM similarity.

In particular, this package includes a model for learning peptide embeddings
such that the embedding of two peptides in the vector space is proportional to
their BLOSUM62 similarity.

### Installation

This project is written in `python3` and can be installed with `pip`.

```
pip3 install .
```