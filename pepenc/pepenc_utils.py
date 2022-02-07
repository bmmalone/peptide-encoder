""" This module contains various helpers for the pepenc project.
"""
import logging
logger = logging.getLogger(__name__)

import torch

def calculate_matched_minkowski_distances(x_1:torch.Tensor, x_2:torch.Tensor, p:float=2.0) -> torch.Tensor:
    """ Calculate the distance between row-wise matched vectors

    This function is only valid for two-dimensional tensors.

    Parameters
    ----------
    x_{1,2} : torch.Tensor
        The two tensors. **N.B.** The tensors must have the same shape (and be located on the same device).

    p : float
        $p$ value for the $p$-norm distance to calculate between the vectors, $p > 0$

    Returns
    -------
    distances : torch.Tensor
        The distance between each pair of row-wise matched vectors. For example, `distances[0] = dist(x_1[0], x_2[0])`.
        Thus, the size of the tensor will be `x_1.shape[0]` (which is the same as `x_2.shape[0]`).
    """
    assert len(x_1.shape) == 2
    assert x_1.shape == x_2.shape

    d = torch.pow(x_1 - x_2, p) # square the element-wise difference
    d = torch.sum(d, 1) # sum for each row
    d = torch.pow(d, 1.0/p) # and take the square root
    return d
