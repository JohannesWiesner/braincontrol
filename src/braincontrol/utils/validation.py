#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for input validation

# FIXME: Split up into validation.py and resolving.py

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd
from nilearn.image import check_niimg

###############################################################################
## Validation helpers for matrices
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

# FIXME: This function should not check if the input is a dataframe, it should
# expect it
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

def _validate_A(A):
    """Validate an adjacency matrix."""
    _validate_square_matrix(A, "A")
    _validate_symmetric_dataframe_labels(A, "A")

def _resolve_A(A):
    """Resolve an adjacency matrix to a NumPy array."""
    return np.asarray(A)

def _validate_B(B):
    """Validate a control input matrix."""
    _validate_square_matrix_or_identity(B, "B")
    _validate_symmetric_dataframe_labels(B, "B")

def _resolve_B(B, n_nodes):
    """Resolve a control input matrix to a NumPy array."""
    return _resolve_array_or_identity(B, n_nodes)

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

def _resolve_S(S, energy_type, n_nodes):
    """Resolve S to the value required by nctpy."""
    if energy_type == "minimal":
        return np.zeros((n_nodes, n_nodes))

    return _resolve_array_or_identity(S, n_nodes)

###############################################################################
## Validation helpers for single parameters
###############################################################################

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

# TODO: Should be renamed _validate_positive_real_number
def _validate_positive_real(value, name):
    """Validate that input is a positive & finite real number."""
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
        
def _validate_rho(rho, energy_type):
    """Validate rho for the selected energy type."""
    if energy_type == "minimal":
        if rho is not None:
            raise ValueError(
                "rho must be None when energy_type='minimal'."
            )

    elif energy_type == "optimal":
        if rho is None:
            raise ValueError(
                "rho must have a value when energy_type='optimal'."
            )

        if not isinstance(rho, (float, np.floating)):
            raise TypeError(
                "rho must be a float."
            )

        _validate_positive_real(rho, "rho")

        if rho > 1.0:
            raise ValueError(
                "rho must be less than or equal to 1."
            )

def _resolve_rho(rho, energy_type):
    """Resolve rho to the value required by nctpy."""
    if energy_type == "minimal":
        # nctpy requires positive rho even when S is zero.
        return 1.0

    return rho

###############################################################################
## Validation helpers for state-like input
###############################################################################

def _is_niimg_like(value):
    """Return True if input can be parsed into a Niimg-like object, otherwise False"""
    try:
        check_niimg(value)
    except (TypeError, ValueError):
        return False
    return True

def _validate_xr(xr, energy_type):
    """Validate the reference state for the selected energy type."""

    if energy_type == "minimal":
        if xr is not None:
            raise ValueError(
                "xr must be None when energy_type='minimal'."
            )

    elif energy_type == "optimal":

        if isinstance(xr, str):
            if xr not in ("zero", "x0", "xf", "midpoint"):
                raise ValueError(
                    "xr must be 'zero', 'x0', 'xf', 'midpoint', "
                    "a single Niimg-like object, or an array-like "
                    "object representing one state."
                )

        elif _is_niimg_like(xr):
            xr_img = check_niimg(
                xr,
                atleast_4d=True,
            )

            if xr_img.shape[3] != 1:
                raise ValueError(
                    "If xr is Niimg-like, it must represent "
                    "exactly one state; "
                    f"got image shape {xr_img.shape}."
                )

        # TODO: We must work on this. My current contract is: If the 
        # object is 1D than rows represent nodes, but if it's 2D then columns
        # represent nodes. So nctpy wants a different shape than braincontrol.
        elif isinstance(xr, np.ndarray):
            if xr.ndim != 2 or xr.shape[1] != 1:
                raise ValueError(
                    f"xr must have shape (N, 1); got {xr.shape}."
                )

            if not np.issubdtype(xr.dtype, np.number):
                raise TypeError(
                    "xr must contain numeric values."
                )

            if not np.all(np.isfinite(xr)):
                raise ValueError(
                    "xr must contain only finite values."
                )

        else:
            raise TypeError(
                "xr must be 'zero', 'x0', 'xf', 'midpoint', "
                "a single Niimg-like object, or a NumPy array "
                "of shape (N, 1) when energy_type='optimal'."
            )

def _validate_transition_states(X=None, X0=None, Xf=None):
    """Validate transition-state inputs.

    Either ``X`` must be provided alone, or ``X0`` and ``Xf`` must
    both be provided.

    Parameters
    ----------
    X : object, optional
        Full sequence of states.
    X0, Xf : object, optional
        Initial and final state sets.

    Raises
    ------
    ValueError
        If the input combination or node labels are invalid.
    """

    # X cannot be combined with X0 or Xf.
    if X is not None and (X0 is not None or Xf is not None):
        raise ValueError(
            "Provide either X or X0 and Xf, not both."
        )

    # At least one transition representation must be provided.
    if X is None and X0 is None and Xf is None:
        raise ValueError(
            "Provide either X or both X0 and Xf."
        )

    # X0 and Xf must always be provided together.
    if X is None and (X0 is None or Xf is None):
        raise ValueError(
            "X0 and Xf must be provided together."
        )

def _validate_state_array(value, name):
    """Validate an array-like state object.

    Parameters
    ----------
    value : array-like
        State object to validate. Supported types include lists, tuples,
        NumPy arrays, pandas Series, and pandas DataFrames. The state-object
        must be either 1D (rows represent nodes) or 2D (columns represent nodes)
    name : str
        Name of the state object used in error messages.

    Raises
    ------
    TypeError
        If the input is not a supported array-like object.
    ValueError
        If the input is not 1D or 2D, or contains non-finite values.
    """
    array_like_types = (
        list,
        tuple,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
    )

    if not isinstance(value, array_like_types):
        raise TypeError(
            f"{name} must be a 1D/2D array-like object."
        )

    state_array = np.asarray(value)

    if state_array.ndim not in (1, 2):
        raise ValueError(
            f"{name} must be 1D or 2D, "
            f"got {state_array.ndim}D with shape "
            f"{state_array.shape}."
        )

    if not np.all(np.isfinite(state_array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )

###############################################################################
## Validation helpers for node-objects (includes matrices and states)
###############################################################################

def _get_node_count(array):
    """Return the number of nodes in a state object.

    Parameters
    ----------
    array : array-like
        State object. For a 1D object, elements represent nodes.
        For a 2D object, columns represent nodes.

    Returns
    -------
    n_nodes : int
        Number of nodes in the state object.
    """

    if array.ndim == 1:
        n_nodes = array.shape[0]
    else:
        n_nodes = array.shape[1]

    return n_nodes

def _validate_node_counts(node_counts):
    """Validate that all defined node counts are equal.

    Parameters
    ----------
    node_counts : dict
        Mapping of node-object names to node counts. Values may be integers
        or None. None values are ignored when comparing node counts.

    Raises
    ------
    ValueError
        If the defined node counts are not all equal.
    """
    counts = {
        name: count
        for name, count in node_counts.items()
        if count is not None
    }

    unique_counts = set(counts.values())

    if len(unique_counts) > 1:
        raise ValueError(
            "All state inputs must have the same number of nodes. "
            f"Got {counts}."
        )

def _get_common_node_count(node_counts):
    """Return the common node count from validated node counts.

    Parameters
    ----------
    node_counts : dict
        Mapping of node-object names to node counts. Values may be integers
        or None. All defined node counts are assumed to be equal.

    Returns
    -------
    n_nodes : int or None
        Common number of nodes, or None if no node count is defined.
    """
    for n_nodes in node_counts.values():
        if n_nodes is not None:
            return n_nodes

    return None

def _get_node_labels(obj):
    """Return node labels if available or None.

    Parameters
    ----------
    obj : object
        Object from which to extract node labels. For a 1D object the node
        labels are the indices, for a 2D object the node labels are the columns

    Returns
    -------
    pandas.Index or None
        Node labels if the object provides them, otherwise None.
    """
    if isinstance(obj, pd.Series):
        return obj.index

    if isinstance(obj, pd.DataFrame):
        return obj.columns

    return None

def _validate_node_labels(node_labels):
    """Validate that all defined node labels are equal.

    Parameters
    ----------
    node_labels : dict
        Mapping of node-object names to node labels. Values may be pandas
        Index objects or None. None values are ignored when comparing labels.

    Raises
    ------
    ValueError
        If any defined node labels do not match.
    """
    labels = {
        name: label
        for name, label in node_labels.items()
        if label is not None
    }

    label_items = list(labels.items())

    for i, (name_a, labels_a) in enumerate(label_items):
        for name_b, labels_b in label_items[i + 1:]:
            if not labels_a.equals(labels_b):
                raise ValueError(
                    "Node labels must match across all inputs. "
                    f"{name_a} and {name_b} have different node labels."
                )

def _get_common_node_labels(node_labels):
    """Return the common node labels from validated node labels.

    Parameters
    ----------
    node_labels : dict
        Mapping of node-object names to node labels. Values may be pandas
        Index objects or None. All defined node labels are assumed to be equal.

    Returns
    -------
    labels : pandas.Index or None
        Common node labels, or None if no node labels are defined.
    """
    for labels in node_labels.values():
        if labels is not None:
            return labels

    return None

# FIXME: Put this logic into _validate_transform_schema
def _validate_transform_node_labels(
    fitted_node_labels,
    transform_node_labels,
):
    """Validate transform node labels against fitted node labels."""

    if fitted_node_labels is None and transform_node_labels is None:
        return

    if fitted_node_labels is None:
        raise ValueError(
            "Node labels were provided during transform, but no "
            "node labels were established during fit."
        )

    if transform_node_labels is None:
        raise ValueError(
            "Node labels were established during fit, but no "
            "node labels were provided during transform."
        )

    if not transform_node_labels.equals(fitted_node_labels):
        raise ValueError(
            "Node labels provided during transform do not match "
            "the node labels established during fit."
        )
        
def _validate_transition_strategy(
    transitions,
    X,
):
    """Validate transition strategy against the state-input configuration."""

    if X is not None:
        valid_transitions = (
            "directed",
            "undirected",
            "directed_with_self",
            "self",
        )
    else:
        valid_transitions = (
            "all_to_all",
            "paired",
        )

    if transitions not in valid_transitions:
        raise ValueError(
            f"transitions={transitions!r} is not compatible "
            "with the provided state inputs. "
            f"Expected one of {valid_transitions}."
        )