#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:49:16 2026

Validation helpers to check schema meta data like

- Number of nodes
- Node labels
- State Labels

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd

###############################################################################
## Validation helpers for node-objects, i.e. states and matrices
###############################################################################

def _get_node_count(array):
    """Return the number of nodes of a state object.

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
    
    # return the first node_count that is not None (at this point we know
    # that all node counts are equal)
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
    
    # return the first node_labels that is not None (at this point we know
    # that all node labels are equal)
    for labels in node_labels.values():
        if labels is not None:
            return labels

    return None

# FIXME: Put this logic into _validate_transform_schema, as it's only used
# once (?)
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
        
###############################################################################
## Validation helpers for state labels
###############################################################################

def _get_state_labels(obj):
    """Return state labels if available or None.

    Parameters
    ----------
    obj : object
        State object from which to extract state labels. For a pandas Series,
        the name identifies its single state. For a pandas DataFrame, the
        index identifies its states.

    Returns
    -------
    pandas.Index or None
        State labels if available, otherwise None.
    """
    if isinstance(obj, pd.Series):
        if obj.name is None:
            return None

        return pd.Index([obj.name])

    if isinstance(obj, pd.DataFrame):
        return obj.index

    return None

def _validate_state_labels(state_labels):
    """Validate state-label consistency across state inputs.

    Parameters
    ----------
    state_labels : dict
        Mapping of state-input names to state labels. For separate initial
        and final state inputs, X0 and Xf must either both provide state
        labels or both be unlabeled. If labeled, their index structures
        must be compatible.

    Raises
    ------
    ValueError
        If only one of X0 and Xf provides state labels, or if their
        index structures are incompatible.
    """
    X0_labels = state_labels["X0"]
    Xf_labels = state_labels["Xf"]

    # X0 and Xf must either both be labeled or both be unlabeled.
    if (X0_labels is None) != (Xf_labels is None):
        raise ValueError(
            "X0 and Xf must either both provide state labels "
            "or both be unlabeled."
        )

    # Nothing else needs to be checked if X0 and Xf are unlabeled.
    if X0_labels is None:
        return

    X0_is_multiindex = isinstance(
        X0_labels,
        pd.MultiIndex,
    )
    Xf_is_multiindex = isinstance(
        Xf_labels,
        pd.MultiIndex,
    )

    # X0 and Xf must use the same kind of index.
    if X0_is_multiindex != Xf_is_multiindex:
        raise ValueError(
            "X0 and Xf state labels must use compatible index "
            "structures. Both must be either Index or MultiIndex."
        )

    # For MultiIndex labels, the level structure must match.
    if X0_is_multiindex:
        if X0_labels.nlevels != Xf_labels.nlevels:
            raise ValueError(
                "X0 and Xf state-label MultiIndexes must have "
                "the same number of levels."
            )

        if X0_labels.names != Xf_labels.names:
            raise ValueError(
                "X0 and Xf state-label MultiIndexes must have "
                "the same level names."
            )

# TODO: This was generated by ChatGTP and I don't find this readable.
def _get_transition_labels(
    state_labels,
    transition_indices,
    n_initial_states=None,
):
    """Return labels for the requested state transitions.

    Parameters
    ----------
    state_labels : dict
        Mapping of state-input names to state labels.

    transition_indices : list of tuple
        State transitions represented as
        ``(transition, source, target)``.

    n_initial_states : int or None, default=None
        Number of initial states when separate X0 and Xf inputs are used.
        If None, transitions are assumed to be defined within a single
        state set X.

    Returns
    -------
    pandas.Index or pandas.MultiIndex or None
        Labels identifying the requested state transitions, or None if
        the states are unlabeled.
    """
    # Transitions within a single state set.
    # TODO: Feels a bit weird to still define source and target labels
    # when just X is provided
    if n_initial_states is None:
        source_labels = state_labels["X"]
        target_labels = state_labels["X"]
        target_offset = 0

    # Transitions from X0 to Xf.
    else:
        source_labels = state_labels["X0"]
        target_labels = state_labels["Xf"]
        target_offset = n_initial_states

    # State-label validation guarantees that X0 and Xf are either both
    # labeled or both unlabeled.
    if source_labels is None:
        return None

    # Preserve the individual levels of a MultiIndex.
    if isinstance(source_labels, pd.MultiIndex):
        transition_values = []

        for level in range(source_labels.nlevels):
            source_values = source_labels.get_level_values(
                level
            ).to_numpy()

            target_values = target_labels.get_level_values(
                level
            ).to_numpy()

            level_transitions = [
                (
                    source_values[source],
                    target_values[target - target_offset],
                )
                for _, source, target in transition_indices
            ]

            transition_values.append(
                level_transitions
            )

        return pd.MultiIndex.from_arrays(
            transition_values,
            names=source_labels.names,
        )

    # For a regular Index, represent each transition as a
    # (source, target) tuple.
    transition_values = [
        (
            source_labels[source],
            target_labels[target - target_offset],
        )
        for _, source, target in transition_indices
    ]

    # Keep each (source, target) tuple as one scalar Index value.
    values = np.empty(
        len(transition_values),
        dtype=object,
    )
    values[:] = transition_values

    return pd.Index(
        values,
        name="transition",
    )