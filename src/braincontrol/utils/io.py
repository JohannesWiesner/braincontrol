#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for input-output operations

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd
import xarray as xr

# TODO: We can deprecate this for now. It's only needed when we allow user
# to pass state label themselves. Only in this case we would need this function
# because only then we would need to check that the user state labels input 
# is valid and can be parsed into a index object

# from collections.abc import Mapping

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
    trajectories,
    *,
    node_labels=None,
    transition_labels=None,
    name=None,
):
    """Return trajectories as a labelled xarray DataArray.

    Parameters
    ----------
    trajectories : list of ndarray or None
        Trajectories for all state transitions. Each element must have
        shape ``(n_time_points, n_nodes)``. All trajectories must have
        the same shape.

    node_labels : pandas.Index or pandas.MultiIndex, optional
        Labels for the node dimension. If None, xarray uses integer
        positions for the node dimension.

    transition_labels : pandas.Index or pandas.MultiIndex, optional
        Labels for the transition dimension. If None, xarray uses integer
        positions for the transition dimension.

    name : str, optional
        Name of the returned DataArray.

    Returns
    -------
    xarray.DataArray or None
        Trajectories with dimensions ``("time", "node", "transition")``.
        Returns None if ``trajectories`` is None.
    """
    if trajectories is None:
        return None

    # Stack trajectories along the transition dimension.
    array = np.stack(
        trajectories,
        axis=2,
    )

    # Add time coordinates.
    coords = {
        "time": np.arange(array.shape[0]),
    }

    # Add node coordinates.
    if node_labels is not None:
        if isinstance(node_labels, pd.MultiIndex):
            coords.update(
                xr.Coordinates.from_pandas_multiindex(
                    node_labels,
                    dim="node",
                )
            )
        else:
            coords["node"] = (
                "node",
                node_labels.to_numpy(),
            )

    # Add transition coordinates.
    if transition_labels is not None:
        if isinstance(transition_labels, pd.MultiIndex):
            coords.update(
                xr.Coordinates.from_pandas_multiindex(
                    transition_labels,
                    dim="transition",
                )
            )
        else:
            coords["transition"] = (
                "transition",
                transition_labels.to_numpy(),
            )

    return xr.DataArray(
        array,
        dims=("time", "node", "transition"),
        coords=coords,
        name=name,
    )