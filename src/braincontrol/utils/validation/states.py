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
    X0,
    Xf,
):
    """Validate the transition strategy for the processed state inputs.

    The valid transition strategies depend on how states are provided and
    on the number of states represented by those inputs.

    When states are provided through a single state set ``X``:

    - If ``X`` contains a single state, ``"self"`` is the only valid
      transition strategy.
    - If ``X`` contains multiple states, ``"directed"``, ``"undirected"``,
      ``"directed_with_self"``, and ``"self"`` are valid strategies.
    - ``"directed"`` computes all directed transitions between distinct
      states.
    - ``"undirected"`` computes one transition for each pair of distinct
      states.
    - ``"directed_with_self"`` computes all directed transitions,
      including self-transitions.
    - ``"self"`` computes one self-transition for each state.

    When states are provided through separate initial and final state sets
    ``X0`` and ``Xf``:

    - ``"all_to_all"`` computes every transition from a state in ``X0``
      to a state in ``Xf``.
    - ``"paired"`` computes transitions between corresponding states in
      ``X0`` and ``Xf`` and therefore requires both inputs to contain the
      same number of states.

    This function assumes that the X / X0 / Xf input configuration has
    already been validated by :func:`_validate_transition_states` and that
    the state inputs have already been converted to their processed NumPy
    representations.

    Parameters
    ----------
    transitions : str
        Transition strategy to validate.

    X : ndarray or None
        Processed single-set state input. A one-dimensional array represents
        one state. For a two-dimensional array, rows represent states.

    X0 : ndarray or None
        Processed initial-state input. A one-dimensional array represents
        one state. For a two-dimensional array, rows represent states.

    Xf : ndarray or None
        Processed final-state input. A one-dimensional array represents
        one state. For a two-dimensional array, rows represent states.

    Raises
    ------
    ValueError
        If ``transitions`` is incompatible with the provided state inputs,
        if a single state in ``X`` is used with a strategy other than
        ``"self"``, or if ``transitions="paired"`` is requested with
        different numbers of initial and final states.
    """

    # Validate transitions within a single state set.
    if X is not None:
        if X.ndim == 1:
            n_states = 1
        else:
            n_states = X.shape[0]

        if n_states == 1:
            valid_transitions = (
                "self",
            )
        else:
            valid_transitions = (
                "directed",
                "undirected",
                "directed_with_self",
                "self",
            )

        if transitions not in valid_transitions:
            raise ValueError(
                f"transitions={transitions!r} is not compatible "
                f"with {n_states} state(s) provided through X. "
                f"Expected one of {valid_transitions}."
            )

        return

    # Validate transitions between initial and final state sets.
    if X0.ndim == 1:
        n_initial_states = 1
    else:
        n_initial_states = X0.shape[0]

    if Xf.ndim == 1:
        n_final_states = 1
    else:
        n_final_states = Xf.shape[0]

    valid_transitions = (
        "all_to_all",
        "paired",
    )

    if transitions not in valid_transitions:
        raise ValueError(
            f"transitions={transitions!r} is not compatible "
            "with states provided through X0 and Xf. "
            f"Expected one of {valid_transitions}."
        )

    if (
        transitions == "paired"
        and n_initial_states != n_final_states
    ):
        raise ValueError(
            "X0 and Xf must contain the same number of states "
            "when transitions='paired'."
        )