#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:40:18 2026

Validation helpers for matrices

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd

# TODO: This function is only used once in _validate_square_matrix so we can
# also just put everything inside there
def _validate_2d_matrix_and_finite(value, name):
    """Validate that input is two-dimensional and contains only finite values."""
    
    # TODO: Should break here if conversion to array is not possible?
    array = np.asarray(value)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be two-dimensional"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must contain only finite values"
        )

def _validate_square_matrix(value, name):
    """Validate that input is a finite square NumPy array or DataFrame."""
    if not isinstance(value, (np.ndarray, pd.DataFrame)):
        raise TypeError(
            f"{name} must be a NumPy array or pandas DataFrame"
        )

    _validate_2d_matrix_and_finite(
        value,
        name,
    )

    if value.shape[0] != value.shape[1]:
        raise ValueError(
            f"{name} must be square; got shape {value.shape}"
        )

def _validate_square_matrix_or_identity(value, name):
    """Validate a finite square matrix or the string ``'identity'``."""
    if isinstance(value, str):
        if value != "identity":
            raise ValueError(
                f"{name} must be a square matrix or 'identity'"
            )
        return

    _validate_square_matrix(
        value,
        name,
    )

# FIXME: This function should not check if the input is a dataframe, it should
# already expect it (?)
def _validate_symmetric_dataframe_labels(obj, name):
    """Validate that row and column indices of a symmetric matrix DataFrame
    are the same.

    Parameters
    ----------
    obj : object
        Matrix to validate. Non-DataFrame objects are ignored.
    name : str
        Name of the matrix used in error messages.

    Raises
    ------
    ValueError
        If the matrix is a DataFrame and its index and column labels
        do not match.
    """
    if isinstance(obj, pd.DataFrame):
        if not obj.index.equals(obj.columns):
            raise ValueError(
                f"{name} must have matching index and column labels."
            )

def _validate_A(A):
    """Validate an adjacency matrix."""
    _validate_square_matrix(A, "A")
    _validate_symmetric_dataframe_labels(A, "A")

def _validate_B(B):
    """Validate a control input matrix."""
    _validate_square_matrix_or_identity(B, "B")
    _validate_symmetric_dataframe_labels(B, "B")

def _validate_S(S, energy_type):
    """Validate S for the selected energy type."""
    if energy_type == "minimal":
        if S is not None:
            raise ValueError(
                "S must be None when energy_type='minimal'."
            )

    elif energy_type == "optimal":
        if S is None:
            raise ValueError(
                "S must have a value when energy_type='optimal'."
            )

        _validate_square_matrix_or_identity(S, "S")
        _validate_symmetric_dataframe_labels(S, "S")

# TODO: I think this is only used for matrices so we can name this function
# more explicit, i.e. _validate_same_shape_matrices
def _validate_same_shape(arrays, names):
    """Validate that all arrays have the same shape."""
    shapes = {
        array.shape
        for array in arrays
    }

    if len(shapes) != 1:
        formatted_shapes = ", ".join(
            f"{name}: {array.shape}"
            for name, array in zip(names, arrays)
        )

        raise ValueError(
            f"{', '.join(names)} must have the same shape; "
            f"got {formatted_shapes}"
        )