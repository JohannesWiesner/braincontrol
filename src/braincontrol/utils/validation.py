#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for input validation

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd
from nilearn.image import check_niimg
from nilearn.image import concat_imgs
from typing import get_args
from nilearn.nilearn_typing import NiimgLike

###############################################################################
## Validation helpers for Network Control Theory parameters
###############################################################################

def _validate_2d_matrix_and_finite(value, name):
    """Validate that input is two-dimensional and contains only finite values."""
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

def _resolve_array_or_identity(value, n_nodes):
    """Resolve a prevalidated matrix or ``'identity'`` to a NumPy array."""
    if isinstance(value, str):
        return np.eye(n_nodes)

    return np.asarray(value)

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

def _validate_choice(value, name, choices):
    """Validate an enumerated option."""
    if value not in choices:
        formatted_choices = ", ".join(
            repr(choice)
            for choice in choices
        )

        raise ValueError(
            f"{name} must be one of {formatted_choices}; "
            f"got {value!r}"
        )

def _validate_boolean(value, name):
    """Validate a boolean option."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a boolean"
        )

def _validate_positive_real(value, name):
    """Validate that input is a positive finite real number."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    ):
        raise TypeError(
            f"{name} must be a real number"
        )

    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be a positive finite number"
        )

def _validate_rho(rho):
    """Validate the mixing parameter for optimal control."""
    if (
        isinstance(rho, (bool, np.bool_))
        or not isinstance(rho, (float, np.floating))
    ):
        raise TypeError(
            "rho must be a float"
        )

    if not np.isfinite(rho) or not 0 < rho <= 1:
        raise ValueError(
            "rho must be a positive finite float between 0 and 1"
        )

def _validate_time_horizon(T, system):
    """Validate the control time horizon for the selected system."""
    if isinstance(T, (bool, np.bool_)) or not np.isscalar(T):
        raise TypeError(
            "T must be a scalar number"
        )

    if not np.isfinite(T) or T <= 0:
        raise ValueError(
            "T must be a positive finite number"
        )

    if system == "discrete":
        if not isinstance(T, (int, np.integer)) or T < 2:
            raise ValueError(
                "T must be an integer of at least 2 "
                "for a discrete system"
            )

    elif not isinstance(T, (float, np.floating)):
        raise TypeError(
            "T must be a float for a continuous system"
        )

###############################################################################
## Validation helpers to check state input(s)
###############################################################################

def _is_niimg_or_tabular_like(value):
    """Determine whether input is Niimg-like or tabular-like.

    Niimg-like input includes:

    - Any individual object whose type is included in Nilearn's
      ``NiimgLike`` type alias, such as a NiBabel spatial image or a
      supported image path.
    - A non-empty list, tuple, or pandas Series whose elements are all
      Niimg-like objects.

    Tabular-like input includes:

    - NumPy arrays.
    - pandas DataFrames.
    - pandas Series.
    - Lists or tuples that are not recognized as collections of
      Niimg-like objects.

    Collections of Niimg-like objects are validated together with
    :func:`nilearn.image.check_niimg`. Consequently, genuine image
    validation errors, such as incompatible shapes or fields of view,
    are propagated rather than treating the input as tabular.

    Parameters
    ----------
    value : object
        Input to classify.

    Returns
    -------
    {"niimg_like", "tabular_like"}
        Classification of the input.

    Raises
    ------
    TypeError
        If the input is neither Niimg-like nor tabular-like.
    ValueError
        If the input is Niimg-like but fails Nilearn's image validation.
    """
    
    # gets you all valid niimg-like data types
    niimg_types = get_args(NiimgLike)

    # Single Niimg-like object.
    if isinstance(value, niimg_types):
        check_niimg(value)
        return "niimg_like"

    # Collection input: either a collection of niimg-like or tabular-like.
    if isinstance(value, (list, tuple, pd.Series)):
        if len(value) > 0 and all(
            isinstance(item, niimg_types)
            for item in value
        ):
            check_niimg(list(value))
            return "niimg_like"

        return "tabular_like"

    # Other tabular input.
    if isinstance(value, (np.ndarray, pd.DataFrame)):
        return "tabular_like"

    raise TypeError(
        "Input must be a Niimg-like or tabular-like object"
    )

# TODO: What about the atleast4 option? 
# https://nilearn.github.io/dev/modules/generated/nilearn.image.check_niimg.html#nilearn.image.check_niimg
def _validate_single_state_niimg(value, name):
    """Validate that Niimg-like input represents exactly one state."""
    img = check_niimg(value)

    if img.ndim == 3:
        return img

    if img.ndim == 4 and img.shape[3] == 1:
        return img

    raise ValueError(
        f"{name} must represent exactly one state; "
        f"got image shape {img.shape}"
    )

# TODO: Maybe this could also already output the output type then we would not
# have to call _is_niimg_or_tabular_like later again
def _resolve_state_input(X=None, x0=None, xf=None):
    """Validate state input and return it in a consistent representation.

    States are represented by rows and nodes by columns for tabular input,
    and by volumes along the fourth dimension for Niimg-like input.

    Exactly one of the following must be provided:

    - ``X``: a two-dimensional NumPy array, pandas DataFrame, or Niimg-like
      object containing one or more states.
    - ``x0`` and ``xf``: two one-dimensional NumPy arrays, two pandas Series,
      or two Niimg-like objects representing one state each.

    Tabular input is returned as a DataFrame with shape
    ``(n_states, n_nodes)``. Niimg-like input is returned as a 4D Niimg-like
    object with one volume per state.

    Parameters
    ----------
    X : np.ndarray, pd.DataFrame, or Niimg-like, optional
        State input. For tabular input, rows represent states and columns
        represent nodes. Niimg-like input may contain one or more states.
    x0 : np.ndarray, pd.Series, or Niimg-like, optional
        Initial state. Niimg-like input must represent exactly one state.
    xf : np.ndarray, pd.Series, or Niimg-like, optional
        Final state. Niimg-like input must represent exactly one state.

    Returns
    -------
    pd.DataFrame or Niimg-like
        Resolved state input. Tabular input is returned as a DataFrame.
        Niimg-like input is returned as a 4D image.
    """
    endpoints_provided = x0 is not None or xf is not None

    if X is not None and endpoints_provided:
        raise ValueError("Provide either X or x0 and xf, not both")

    if X is None and x0 is None and xf is None:
        raise ValueError("Provide either X or both x0 and xf")

    # X contains all states
    if X is not None:
        X_type = _is_niimg_or_tabular_like(X)

        if X_type == "niimg_like":
            # concat_imgs guarantees a 4D representation.
            return concat_imgs([X])

        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError(
                "X must be a NumPy array, pandas DataFrame, "
                "or Niimg-like object"
            )

        _validate_2d_matrix_and_finite(X, "X")

        if isinstance(X, pd.DataFrame):
            return X.copy()

        return pd.DataFrame(X)

    # x0 and xf must be provided together
    if x0 is None or xf is None:
        raise ValueError("x0 and xf must be provided together")

    x0_type = _is_niimg_or_tabular_like(x0)
    xf_type = _is_niimg_or_tabular_like(xf)

    if x0_type != xf_type:
        raise TypeError("x0 and xf must use the same input type")

    # Niimg-like endpoints
    if x0_type == "niimg_like":
        x0_img = _validate_single_state_niimg(x0, "x0")
        xf_img = _validate_single_state_niimg(xf, "xf")

        return concat_imgs([x0_img, xf_img])

    # Tabular endpoints must use the same concrete representation.
    if type(x0) is not type(xf):
        raise TypeError("x0 and xf must use the same input type")

    if isinstance(x0, np.ndarray):
        if x0.ndim != 1 or xf.ndim != 1:
            raise ValueError(
                "x0 and xf must be one-dimensional states"
            )

        if x0.shape != xf.shape:
            raise ValueError(
                "x0 and xf must contain the same number of nodes"
            )

        if x0.dtype != xf.dtype:
            raise TypeError(
                "x0 and xf must have the same dtype"
            )

        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(xf)):
            raise ValueError(
                "x0 and xf must contain only finite values"
            )

        return pd.DataFrame(
            np.stack((x0, xf))
        )

    if isinstance(x0, pd.Series):
        if not x0.index.equals(xf.index):
            raise ValueError(
                "x0 and xf must have matching indices"
            )

        if x0.dtype != xf.dtype:
            raise TypeError(
                "x0 and xf must have the same dtype"
            )

        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(xf)):
            raise ValueError(
                "x0 and xf must contain only finite values"
            )

        return pd.DataFrame(
            [x0.to_numpy(), xf.to_numpy()],
            columns=x0.index,
        )

    raise TypeError(
        "x0 and xf must both be NumPy arrays, both be pandas Series, "
        "or both be Niimg-like objects"
    )
    
    
def _validate_transition_order(n_states, order):
    """Validate that the requested transition order is possible."""
    order = _validate_choice(
        order,
        "order",
        ("combinations", "permutations", "product", "stability"),
    )

    if n_states < 1:
        raise ValueError("State input must contain at least one state")

    if n_states == 1 and order in ("combinations", "permutations"):
        raise ValueError(
            f"order={order!r} requires at least two states"
        )

    return order