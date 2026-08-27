#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for input-output operations

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd
from collections.abc import Mapping
from .validation import _validate_choice
from itertools import combinations, permutations, product
import xarray as xr

# TODO: Find more suitable names for order choices
def _set_transition_order(n_states, order):
    """Return the number and indices of requested state transitions.

    Each transition tuple contains ``(transition, source, target)``.  The
    ordering matches the corresponding iterator in :mod:`itertools`.
    """
    
    if not isinstance(n_states, (int, np.integer)) or isinstance(n_states, bool):
        raise TypeError("n_states must be an integer")
    if n_states < 1:
        raise ValueError("n_states must be at least 1")
        
    indices = range(n_states)
    
    if order == "permutations":
        pairs = permutations(indices, r=2)
    elif order == "combinations":
        pairs = combinations(indices, r=2)
    elif order == "product":
        pairs = product(indices, repeat=2)
    else:
        pairs = ((index, index) for index in indices)

    transition_indices = [
        (transition, source, target)
        for transition, (source, target) in enumerate(pairs)
    ]
    
    n_transitions = len(transition_indices)
    
    return n_transitions, transition_indices

def _coerce_labels(labels, expected_length, parameter_name):
    """Convert and validate labels as a pandas Index or MultiIndex.

    List-like input is converted to a named Index. Existing Index names are
    preserved. MultiIndex input must have a name for every level. The number
    of labels must match ``expected_length`` when provided.
    """
    
    if isinstance(labels, (str, bytes, Mapping, pd.DataFrame)):
        raise TypeError(
            f"{parameter_name} must be a list-like, Index, or MultiIndex object"
        )

    if isinstance(labels, pd.MultiIndex):
        index = labels.copy()

        if any(name is None for name in index.names):
            raise ValueError(
                f"All levels of {parameter_name} must have a name"
            )

    elif isinstance(labels, pd.Index):
        index = labels.copy()

    else:
        try:
            values = list(labels)
        except TypeError as error:
            raise TypeError(
                f"{parameter_name} must be a list-like, Index, "
                "or MultiIndex object"
            ) from error

        default_name = {
            "state_labels": "state",
            "node_labels": "node",
        }.get(parameter_name, parameter_name)

        index = pd.Index(values, name=default_name)

    if len(index) == 0:
        raise ValueError(
            f"{parameter_name} must contain at least one value"
        )

    if expected_length is not None and len(index) != expected_length:
        raise ValueError(
            f"{parameter_name} must contain {expected_length} values; "
            f"got {len(index)}"
        )

    return index

def _state_transition_index(state_labels, order):
    """Build an index describing state transitions.

    Each label is replaced by a ``(source, target)`` pair according to
    ``order``. For MultiIndex input, the original levels and their names are
    preserved.
    """
    n_states = len(state_labels)
    _, transition_indices = _set_transition_order(
        n_states,
        order,
    )

    if isinstance(state_labels, pd.MultiIndex):
        level_names = state_labels.names
        level_values = [
            state_labels.get_level_values(level).to_numpy()
            for level in range(state_labels.nlevels)
        ]

        transition_values = [
            [
                (values[source], values[target])
                for _, source, target in transition_indices
            ]
            for values in level_values
        ]

        return pd.MultiIndex.from_arrays(
            transition_values,
            names=level_names,
        )

    transition_values = [
        (state_labels[source], state_labels[target])
        for _, source, target in transition_indices
    ]

    # Keep each (source, target) tuple as one scalar Index value.
    values = np.empty(len(transition_values), dtype=object)
    values[:] = transition_values

    return pd.Index(
        values,
        name=state_labels.name,
    )


def _get_trajectory_array(
    array,
    *,
    node_labels=None,
    transition_labels=None,
    name=None,
):
    """Return a trajectory array as a labelled xarray DataArray.

    Parameters
    ----------
    array : ndarray of shape (n_time_points, n_nodes, n_transitions)
        Trajectory values.
    node_labels : pd.Index or pd.MultiIndex, optional
        Labels for the node dimension.
    transition_labels : pd.Index or pd.MultiIndex, optional
        Labels for the transition dimension.
    name : str, optional
        Name of the returned DataArray.

    Returns
    -------
    xarray.DataArray or None
        Labelled trajectory array, or ``None`` if ``array`` is ``None``.
    """
    if array is None:
        return None

    coords = {
        "time": np.arange(array.shape[0]),
    }

    if node_labels is not None:
        if isinstance(node_labels, pd.MultiIndex):
            coords.update(
                xr.Coordinates.from_pandas_multiindex(
                    node_labels,
                    dim="node",
                )
            )
        else:
            coords["node"] = node_labels

    if transition_labels is not None:
        if isinstance(transition_labels, pd.MultiIndex):
            coords.update(
                xr.Coordinates.from_pandas_multiindex(
                    transition_labels,
                    dim="transition",
                )
            )
        else:
            coords["transition"] = transition_labels

    return xr.DataArray(
        array.copy(),
        dims=("time", "node", "transition"),
        coords=coords,
        name=name,
    )