#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:40:18 2026

@author: johannes.wiesner
"""

import numpy as np

###############################################################################
## Helpers to resolve matrix input
###############################################################################

def _resolve_array_or_identity(value, n_nodes):
    """Resolve a prevalidated matrix or ``'identity'`` to a NumPy array."""
    if isinstance(value, str):
        return np.eye(n_nodes)

    return np.asarray(value)

def _resolve_A(A):
    """Resolve an adjacency matrix to a NumPy array."""
    return np.asarray(A)

def _resolve_B(B, n_nodes):
    """Resolve a control input matrix to a NumPy array."""
    return _resolve_array_or_identity(B, n_nodes)

def _resolve_S(S, energy_type, n_nodes):
    """Resolve S to the value required by nctpy."""
    if energy_type == "minimal":
        return np.zeros((n_nodes, n_nodes))

    return _resolve_array_or_identity(S, n_nodes)

###############################################################################
## Helpers to resolve single value inputs
###############################################################################

def _resolve_rho(rho, energy_type):
    """Resolve rho to the value required by nctpy."""
    if energy_type == "minimal":
        # nctpy requires positive rho even when S is zero.
        return 1.0

    return rho