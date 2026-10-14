#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:24:33 2026

Helpers to compute transitions

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd
from itertools import chain, combinations, permutations, product
from nctpy.energies import get_control_inputs, integrate_u
import xarray as xr

def _get_transition_pairs(
    transitions,
    *,
    X=None,
    X0=None,
    Xf=None,
):
    """Generate explicit initial/final states in transition order.

    Inputs must already be normalized and validated by
    ``_validate_transition_states`` and ``_validate_transition_strategy``.

    Parameters
    ----------
    transitions : {"directed", "undirected", "directed_with_self", "self", \
                   "paired", "all_to_all"}
        Strategy used to select state pairs.
    X : ndarray of shape (n_states, n_nodes), optional
        Single state collection for within-set transitions.
    X0, Xf : ndarray of shape (n_states, n_nodes), optional
        Separate initial and final collections for ``paired`` or
        ``all_to_all``. Their state counts may differ for ``all_to_all``.

    Returns
    -------
    initial_states, final_states : ndarray of shape (n_transitions, n_nodes)
        Corresponding rows define one transition. For ``self`` and
        ``paired``, the input arrays are reused without copying.
    transition_indices : ndarray of shape (n_transitions, 2)
        Source and target positions in their original collections, used
        to select state labels without offsets or concatenation.

    Notes
    -----
    Pairs follow input order with the source varying slowest. Directed
    transitions omit the diagonal; undirected transitions retain only
    pairs whose source position is less than the target position.
    """
    if X is not None:
        source_states = target_states = X
    else:
        source_states, target_states = X0, Xf

    if transitions in ("self", "paired"):
        indices = np.arange(source_states.shape[0])
        return source_states, target_states, np.column_stack((indices, indices))

    source_indices = range(source_states.shape[0])
    if transitions == "directed":
        pairs = permutations(source_indices, r=2)
    elif transitions == "undirected":
        pairs = combinations(source_indices, r=2)
    else:  # transitions in ("directed_with_self", "all_to_all")
        pairs = product(source_indices, range(target_states.shape[0]))

    # Consume pairs without retaining a Python tuple for every transition.
    transition_indices = np.fromiter(
        chain.from_iterable(pairs), dtype=np.intp
    ).reshape(-1, 2)
    initial_states = source_states[transition_indices[:, 0]]
    final_states = target_states[transition_indices[:, 1]]
    return initial_states, final_states, transition_indices

# TODO (#26): Make more memory efficient by predefining empty arrays that have
# n_transitions x n_timepoints x n_nodes. 
# TODO (#26): Make more computaionally efficient by using parallelization.
def _get_transition_trajectories(
    A,
    initial_states,
    final_states,
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

    initial_states, final_states : ndarray of shape (n_transitions, n_nodes)
        Initial and final state of each transition. The arrays must have
        the same shape, with corresponding rows defining one transition.

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

    n_transitions = initial_states.shape[0]

    state_trajectories = []
    control_trajectories = []
    errors = np.zeros((n_transitions, 2))

    for transition, (initial_state, final_state) in enumerate(
        zip(initial_states, final_states)
    ):
        (
            state_trajectory,
            control_trajectory,
            error,
        ) = get_control_inputs(
            A_norm=A,
            T=T,
            B=B,
            x0=initial_state,
            xf=final_state,
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
