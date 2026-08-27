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
        
def _resolve_energy_type_parameters(
    rho,
    S,
    energy_type,
    n_nodes,
):
    """Validate and resolve parameters that depend on energy type.

    For minimal-energy control, ``rho`` and ``S`` must both be ``None``.
    They are resolved to the values required internally by nctpy.

    For optimal control, ``rho`` and ``S`` must both be provided and valid.

    Parameters
    ----------
    rho : float or None
        Mixing parameter.
    S : array-like, "identity", or None
        State-trajectory constraint matrix.
    energy_type : {"minimal", "optimal"}
        Type of control energy.
    A : ndarray of shape (n_nodes, n_nodes)
        Resolved adjacency matrix.
    n_nodes : int
        Number of network nodes.

    Returns
    -------
    rho : float
        Resolved mixing parameter.
    S : ndarray of shape (n_nodes, n_nodes)
        Resolved state-trajectory constraint matrix.
    """
    
    if energy_type == "minimal":
        if rho is not None or S is not None:
            raise ValueError(
                "rho and S must both be None when "
                "energy_type='minimal'"
            )

        # nctpy requires a positive rho internally even when S is zero.
        rho = 1.0
        S = np.zeros((n_nodes,n_nodes))

    elif energy_type == "optimal":
        if rho is None or S is None:
            raise ValueError(
                "rho and S must both be provided when "
                "energy_type='optimal'"
            )

        _validate_rho(rho)
        _validate_square_matrix_or_identity(S,"S",)
        S = _resolve_array_or_identity(S,n_nodes,)

    return rho, S


def _validate_reference_state(value, n_nodes):
    """Validate a reference state accepted by ``nctpy``.

    Reference states may be one of the names understood by ``nctpy``, a
    one-dimensional vector, a column vector, or a three-dimensional
    Niimg-like object. Image data are converted separately, after a masker has
    been fitted.
    """
    choices = ("x0", "xf", "zero", "midpoint")

    if isinstance(value, str):
        _validate_choice(value, "xr", choices)
        return

    if isinstance(value, np.ndarray):
        if value.shape not in ((n_nodes,), (n_nodes, 1)):
            raise ValueError(
                "xr must have shape "
                f"({n_nodes},) or ({n_nodes}, 1); got {value.shape}"
            )

        if not np.issubdtype(value.dtype, np.number):
            raise TypeError("xr must contain numeric values")

        if not np.all(np.isfinite(value)):
            raise ValueError("xr must contain only finite values")

        return

    try:
        image = check_niimg(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "xr must be 'x0', 'xf', 'zero', 'midpoint', a NumPy "
            "reference vector, or a 3D Niimg-like object"
        ) from error

    if len(image.shape) != 3:
        raise ValueError(
            "Image-like xr must be three-dimensional; "
            f"got image shape {image.shape}"
        )


def _resolve_reference_state(value, n_nodes, masker=None):
    """Return a validated reference state in the form required by ``nctpy``."""
    _validate_reference_state(value, n_nodes)

    if isinstance(value, str):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.reshape(n_nodes, 1).copy()

    if masker is None:
        raise ValueError(
            "Image-like xr requires a masker, for example "
            "NiftiLabelsMasker(labels_img=atlas)"
        )

    reference = np.asarray(masker.transform(check_niimg(value)))
    if reference.shape not in ((n_nodes,), (1, n_nodes)):
        raise ValueError(
            "Image-like xr must resolve to the same number of nodes as A; "
            f"expected {n_nodes} nodes, got transformed shape "
            f"{reference.shape}"
        )

    if not np.all(np.isfinite(reference)):
        raise ValueError("Masked xr must contain only finite values")

    return reference.reshape(n_nodes, 1)

###############################################################################
## Validation helpers to check state input(s)
###############################################################################

def _is_niimg_or_tabular_like(value):
    """Determine whether input is Niimg-like or tabular-like.

    Niimg-like input includes:

    - Any individual object whose type is included in Nilearn's ``NiimgLike`` type alias.
    - A non-empty list, tuple, or pandas Series whose elements are all Niimg-like objects.

    Tabular-like input includes:

    - NumPy arrays.
    - pandas DataFrames.
    - pandas Series that are not collections of Niimg-like objects.
    - Lists or tuples that are not collections of Niimg-like objects.

    Niimg-like input is validated with
    :func:`nilearn.image.check_niimg`. Genuine image-validation errors,
    such as incompatible shapes or fields of view, are propagated rather
    than treating the input as tabular-like.

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
        If Niimg-like input fails Nilearn validation.
    """
    niimg_types = get_args(NiimgLike)

    # Single Niimg-like object.
    if isinstance(value, niimg_types):
        check_niimg(value)
        return "niimg_like"

    # Collection input.
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

def _resolve_single_state_niimg(value, name):
    """Resolve Niimg-like input representing exactly one state.

    Three-dimensional input is converted to a singleton 4D image.
    Existing 4D input must contain exactly one volume.

    Parameters
    ----------
    value : Niimg-like
        Image representing a single state.
    name : str
        Parameter name used in error messages.

    Returns
    -------
    Niimg-like
        Validated 4D image containing exactly one state.

    Raises
    ------
    ValueError
        If the image contains more than one state.
    """
    img = check_niimg(
        value,
        atleast_4d=True,
    )

    if img.shape[3] != 1:
        raise ValueError(
            f"{name} must represent exactly one state; "
            f"got image shape {img.shape}"
        )

    return img

# NOTE: Might be smart to split this up in the future for readability reasons
def _resolve_state_input(X=None, x0=None, xf=None):
    """Validate and resolve state input.

    States are represented by rows and nodes by columns for tabular input,
    and by volumes along the fourth dimension for Niimg-like input.

    Exactly one of the following must be provided:

    - ``X`` containing one or more states.
    - ``x0`` and ``xf`` containing exactly one state each.

    Tabular input may be provided as a NumPy array, pandas DataFrame,
    pandas Series, list, or tuple. Lists, tuples, and Series containing
    exclusively Niimg-like objects are instead treated as Niimg-like
    collections.

    Tabular input is returned as a pandas DataFrame with shape
    ``(n_states, n_nodes)``. Niimg-like input is returned as a 4D image with
    one volume per state.

    Parameters
    ----------
    X : array-like, DataFrame, or Niimg-like, optional
        State input containing one or more states.
    x0 : array-like, Series, or Niimg-like, optional
        Initial state.
    xf : array-like, Series, or Niimg-like, optional
        Final state.

    Returns
    -------
    X_resolved : pd.DataFrame, Niimg-like or NumPy array
        Resolved state input.
    X_type : {"tabular_like", "niimg_like"}
        Type of the resolved state input.
    """
    
    # check incompatible inputs
    endpoints_provided = x0 is not None or xf is not None

    if X is not None and endpoints_provided:
        raise ValueError(
            "Provide either X or x0 and xf, not both"
        )

    if X is None and x0 is None and xf is None:
        raise ValueError(
            "Provide either X or both x0 and xf"
        )

    # X contains all states.
    if X is not None:
        
        X_type = _is_niimg_or_tabular_like(X)

        # X is niimg-like
        if X_type == "niimg_like":
            if isinstance(X, pd.Series):
                X = X.tolist()

            X_resolved = check_niimg(X,atleast_4d=True)

            return X_resolved, X_type

        # X is dataframe
        if isinstance(X, pd.DataFrame):
            _validate_2d_matrix_and_finite(X,"X")
            
            X_resolved = X.copy()

            return X_resolved, X_type

        # X is array-like
        X_array = np.asarray(X)
        _validate_2d_matrix_and_finite(X_array,"X",)
        X_resolved = X_array.copy()

        return X_resolved, X_type

    # x0 and xf must be provided together.
    if x0 is None or xf is None:
        raise ValueError(
            "x0 and xf must be provided together"
        )

    # x0 and xf must have the same type
    x0_type = _is_niimg_or_tabular_like(x0)
    xf_type = _is_niimg_or_tabular_like(xf)

    if x0_type != xf_type:
        raise TypeError("x0 and xf must use the same input type")

    # Niimg-like endpoints.
    if x0_type == "niimg_like":
        
        x0_img = _resolve_single_state_niimg(x0,"x0")
        xf_img = _resolve_single_state_niimg(xf,"xf")
        X_resolved = concat_imgs([x0_img, xf_img])

        return X_resolved, "niimg_like"

    # Tabular endpoints must use the same concrete representation.
    if type(x0) is not type(xf):
        raise TypeError("x0 and xf must use the same input type")

    # Preserve Series node labels.
    if isinstance(x0, pd.Series):
        if not x0.index.equals(xf.index):
            raise ValueError("x0 and xf must have matching indices")

        if x0.dtype != xf.dtype:
            raise TypeError("x0 and xf must have the same dtype")

        if (
            not np.all(np.isfinite(x0))
            or not np.all(np.isfinite(xf))
        ):
            raise ValueError("x0 and xf must contain only finite values")

        X_resolved = pd.DataFrame(
            [x0.to_numpy(), xf.to_numpy()],
            columns=x0.index,
        )

        return X_resolved, "tabular_like"

    # Resolve NumPy/list/tuple endpoints.
    x0_array = np.asarray(x0)
    xf_array = np.asarray(xf)

    if x0_array.ndim != 1 or xf_array.ndim != 1:
        raise ValueError("x0 and xf must be one-dimensional states")

    if x0_array.shape != xf_array.shape:
        raise ValueError("x0 and xf must contain the same number of nodes")

    if x0_array.dtype != xf_array.dtype:
        raise TypeError("x0 and xf must have the same dtype")

    if (
        not np.all(np.isfinite(x0_array))
        or not np.all(np.isfinite(xf_array))
    ):
        raise ValueError(
            "x0 and xf must contain only finite values"
        )

    X_resolved = np.stack((x0_array, xf_array))

    return X_resolved, "tabular_like"

# FIXME: _validate functions should never return anything
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
