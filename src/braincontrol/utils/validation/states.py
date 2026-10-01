#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:40:18 2026

Validation helpers for state input

@author: johannes.wiesner
"""

import numpy as np
import pandas as pd
from nilearn.image import check_niimg

def _is_niimg_like(value):
    """Return True if input can be parsed into a Niimg-like object, otherwise False"""
    try:
        check_niimg(value)
    except (TypeError, ValueError):
        return False
    return True

def _validate_xr(xr, energy_type):
    """Validate the reference state for the selected energy type."""

    if energy_type == "minimal":
        if xr is not None:
            raise ValueError(
                "xr must be None when energy_type='minimal'."
            )

    elif energy_type == "optimal":

        # we can use _validate_choice here?
        if isinstance(xr, str):
            if xr not in ("zero", "x0", "xf", "midpoint"):
                raise ValueError(
                    "xr must be 'zero', 'x0', 'xf', 'midpoint', "
                    "a single Niimg-like object, or an array-like "
                    "object representing one state."
                )

        elif _is_niimg_like(xr):
            xr_img = check_niimg(
                xr,
                atleast_4d=True,
            )

            if xr_img.shape[3] != 1:
                raise ValueError(
                    "If xr is Niimg-like, it must represent "
                    "exactly one state; "
                    f"got image shape {xr_img.shape}."
                )

        # TODO: We must work on this. My current contract is: If the
        # object is 1D than rows represent nodes, but if it's 2D then columns
        # represent nodes. So nctpy wants a different shape than braincontrol.
        elif isinstance(xr, np.ndarray):
            if xr.ndim != 2 or xr.shape[1] != 1:
                raise ValueError(
                    f"xr must have shape (N, 1); got {xr.shape}."
                )

            if not np.issubdtype(xr.dtype, np.number):
                raise TypeError(
                    "xr must contain numeric values."
                )

            if not np.all(np.isfinite(xr)):
                raise ValueError(
                    "xr must contain only finite values."
                )

        else:
            raise TypeError(
                "xr must be 'zero', 'x0', 'xf', 'midpoint', "
                "a single Niimg-like object, or a NumPy array "
                "of shape (N, 1) when energy_type='optimal'."
            )

def _validate_transition_states(X=None, X0=None, Xf=None):
    """Validate transition-state inputs.

    Answers: Is the X / X0 / Xf configuration valid?

    Either ``X`` must be provided alone, or ``X0`` and ``Xf`` must
    both be provided.

    Parameters
    ----------
    X : object, optional
        Full sequence of states.
    X0, Xf : object, optional
        Initial and final state sets.

    Raises
    ------
    ValueError
        If the input combination or node labels are invalid.
    """

    # X cannot be combined with X0 or Xf.
    if X is not None and (X0 is not None or Xf is not None):
        raise ValueError(
            "Provide either X or X0 and Xf, not both."
        )

    # At least one transition representation must be provided.
    if X is None and X0 is None and Xf is None:
        raise ValueError(
            "Provide either X or both X0 and Xf."
        )

    # X0 and Xf must always be provided together.
    if X is None and (X0 is None or Xf is None):
        raise ValueError(
            "X0 and Xf must be provided together."
        )
    
def _validate_state_array(value, name):
    """Validate an array-like state object.

    Parameters
    ----------
    value : array-like
        State object to validate. Supported types include lists, tuples,
        NumPy arrays, pandas Series, and pandas DataFrames. The state-object
        must be either 1D (rows represent nodes) or 2D (columns represent nodes)
    name : str
        Name of the state object used in error messages.

    Raises
    ------
    TypeError
        If the input is not a supported array-like object.
    ValueError
        If the input is not 1D or 2D, or contains non-finite values.
    """
    array_like_types = (
        list,
        tuple,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
    )

    if not isinstance(value, array_like_types):
        raise TypeError(
            f"{name} must be a 1D/2D array-like object."
        )

    state_array = np.asarray(value)

    if state_array.ndim not in (1, 2):
        raise ValueError(
            f"{name} must be 1D or 2D, "
            f"got {state_array.ndim}D with shape "
            f"{state_array.shape}."
        )

    if not np.all(np.isfinite(state_array)):
        raise ValueError(
            f"{name} must contain only finite values."
        )
        
def _validate_transition_strategy(
    transitions,
    X,
):
    """Validate the transition strategy for the state-input configuration.

    When states are provided as a single set ``X``, the transition strategy
    must describe transitions within that set. When states are provided as
    separate initial and final sets ``X0`` and ``Xf``, the transition strategy
    must describe transitions between those sets.

    Parameters
    ----------
    transitions : str
        Transition strategy to validate.
    X : object or None
        Single-set state input. If not None, states are assumed to be provided
        as ``X``. If None, states are assumed to be provided as separate
        ``X0`` and ``Xf`` inputs.

    Raises
    ------
    ValueError
        If ``transitions`` is not compatible with the state-input
        configuration.
    """

    if X is not None:
        valid_transitions = (
            "directed",
            "undirected",
            "directed_with_self",
            "self",
        )
    else:
        valid_transitions = (
            "all_to_all",
            "paired",
        )

    if transitions not in valid_transitions:
        raise ValueError(
            f"transitions={transitions!r} is not compatible "
            "with the provided state inputs. "
            f"Expected one of {valid_transitions}."
        )