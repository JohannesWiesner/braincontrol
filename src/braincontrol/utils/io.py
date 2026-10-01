#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for input-output operations

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd
from collections.abc import Mapping
import xarray as xr

# TODO: We can deprecate this for now. It's only needed when we allow user
# to pass state label themselves. Only in this case we would need this function
# because only then we would need to check that the user state labels input 
# is valid and can be parsed into a index object
# def _coerce_labels(labels, expected_length, parameter_name):
#     """Convert and validate labels as a pandas Index or MultiIndex.

#     List-like input is converted to a named Index. Existing Index names are
#     preserved. MultiIndex input must have a name for every level. The number
#     of labels must match ``expected_length`` when provided.
#     """
    
#     if isinstance(labels, (str, bytes, Mapping, pd.DataFrame)):
#         raise TypeError(
#             f"{parameter_name} must be a list-like, Index, or MultiIndex object"
#         )

#     if isinstance(labels, pd.MultiIndex):
#         index = labels.copy()

#         if any(name is None for name in index.names):
#             raise ValueError(
#                 f"All levels of {parameter_name} must have a name"
#             )

#     elif isinstance(labels, pd.Index):
#         index = labels.copy()

#     else:
#         try:
#             values = list(labels)
#         except TypeError as error:
#             raise TypeError(
#                 f"{parameter_name} must be a list-like, Index, "
#                 "or MultiIndex object"
#             ) from error

#         default_name = {
#             "state_labels": "state",
#             "node_labels": "node",
#         }.get(parameter_name, parameter_name)

#         index = pd.Index(values, name=default_name)

#     if len(index) == 0:
#         raise ValueError(
#             f"{parameter_name} must contain at least one value"
#         )

#     if expected_length is not None and len(index) != expected_length:
#         raise ValueError(
#             f"{parameter_name} must contain {expected_length} values; "
#             f"got {len(index)}"
#         )

#     return index

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