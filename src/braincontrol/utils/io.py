#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for input-output operations

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd
import xarray as xr

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