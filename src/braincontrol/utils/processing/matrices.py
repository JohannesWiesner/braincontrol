#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:24:18 2026

Helpers to resolve matrix input

@author: johannes.wiesner
"""

import numpy as np

def _get_array_or_identity(value, n_nodes):
    """Resolve a prevalidated matrix or ``'identity'`` to a NumPy array."""
    if isinstance(value, str):
        return np.eye(n_nodes)

    return np.asarray(value)

def _get_A(A):
    """Resolve an adjacency matrix to a NumPy array."""
    return np.asarray(A)

def _get_B(B, n_nodes):
    """Resolve a control input matrix to a NumPy array."""
    return _get_array_or_identity(B, n_nodes)

def _get_S(S, energy_type, n_nodes):
    """Resolve S to the value required by nctpy."""
    if energy_type == "minimal":
        return np.zeros((n_nodes, n_nodes))

    return _get_array_or_identity(S, n_nodes)