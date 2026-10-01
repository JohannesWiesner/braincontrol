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

import pandas as pd

###############################################################################
## Validation helpers for node-objects, i.e. states and matrices
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

# FIXME: Put this logic into _validate_transform_schema, as it's only used
# once?
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
        labels or both be unlabeled.

    Raises
    ------
    ValueError
        If only one of X0 and Xf provides state labels.
    """
    X0_labels = state_labels["X0"]
    Xf_labels = state_labels["Xf"]

    if (X0_labels is None) != (Xf_labels is None):
        raise ValueError(
            "X0 and Xf must either both provide state labels "
            "or both be unlabeled."
        )