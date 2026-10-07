#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:24:33 2026

Helpers to compute transitions

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd
from itertools import combinations, permutations, product
from nctpy.energies import get_control_inputs, integrate_u
import xarray as xr

# TODO: Add more extensive docstring
def _get_transition_indices(
    n_states,
    transitions,
    *,
    n_initial_states=None,
):
    """Return indices for the requested state transitions."""

    # Transitions within a single state set.
    if n_initial_states is None:
        indices = range(n_states)

        if transitions == "directed":
            pairs = permutations(
                indices,
                r=2,
            )

        elif transitions == "undirected":
            pairs = combinations(
                indices,
                r=2,
            )

        elif transitions == "directed_with_self":
            pairs = product(
                indices,
                repeat=2,
            )

        else:  # transitions == "self"
            pairs = (
                (index, index)
                for index in indices
            )

    # Transitions between initial and final state sets.
    else:
        initial_indices = range(
            n_initial_states
        )
        final_indices = range(
            n_initial_states,
            n_states,
        )

        if transitions == "all_to_all":
            pairs = product(
                initial_indices,
                final_indices,
            )

        else:  # transitions == "paired"
            pairs = zip(
                initial_indices,
                final_indices,
            )

    return [
        (transition, source, target)
        for transition, (source, target) in enumerate(pairs)
    ]

# TODO: This function should be placed elsewhere and imported
# TODO (#26): Make more memory efficient by predefining empty arrays that have
# n_transitions x n_timepoints x n_nodes. 
# TODO (#26): Make more computaionally efficient by using parallelization.
def _get_transition_trajectories(
    A,
    X,
    transition_indices,
    T,
    B,
    rho,
    S,
    *,
    system="continuous",
    xr="xf",
    expm_version="scipy",
):
    """Compute state and control trajectories for state transitions.

    Parameters
    ----------
    A : ndarray of shape (n_nodes, n_nodes)
        Adjacency matrix.

    X : ndarray of shape (n_states, n_nodes)
        State matrix.

    transition_indices : list of tuple
        State transitions represented as
        ``(transition, source, target)``.

    T : float or int
        Time horizon.

    B : ndarray of shape (n_nodes, n_nodes)
        Control input matrix.

    rho : float
        State-trajectory constraint parameter.

    S : ndarray of shape (n_nodes, n_nodes)
        State-trajectory constraint matrix.

    system : {"continuous", "discrete"}, default="continuous"
        System dynamics.

    xr : {"xf", "zero"} or ndarray, default="xf"
        Reference state.

    expm_version : {"scipy", "eig"}, default="scipy"
        Matrix exponential implementation.

    Returns
    -------
    state_trajectories : list of ndarray
        State trajectory for each transition.

    control_trajectories : list of ndarray
        Control trajectory for each transition.

    errors : ndarray of shape (n_transitions, 2)
        Numerical errors for all transitions.
    """

    n_transitions = len(transition_indices)

    state_trajectories = []
    control_trajectories = []
    errors = np.zeros((n_transitions, 2))

    for transition, source, target in transition_indices:
        (
            state_trajectory,
            control_trajectory,
            error,
        ) = get_control_inputs(
            A_norm=A,
            T=T,
            B=B,
            x0=X[source],
            xf=X[target],
            system=system,
            rho=rho,
            S=S,
            xr=xr,
            expm_version=expm_version,
        )

        state_trajectories.append(state_trajectory)
        control_trajectories.append(control_trajectory)
        errors[transition] = error

    return (
        state_trajectories,
        control_trajectories,
        errors,
    )

def _get_transition_energy(control_trajectories):
    """Integrate control trajectories for every transition.

    Parameters
    ----------
    control_trajectories : list of ndarray
        Control trajectories for all transitions. Each element has shape
        (n_timepoints, n_nodes).

    Returns
    -------
    energies : ndarray of shape (n_transitions, n_nodes)
        Integrated control energy for each transition and node.
    """

    n_transitions = len(control_trajectories)
    n_nodes = control_trajectories[0].shape[1]

    energies = np.empty(
        (n_transitions, n_nodes)
    )

    for transition, control_trajectory in enumerate(
        control_trajectories
    ):
        energies[transition] = integrate_u(
            control_trajectory
        )

    return energies

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
