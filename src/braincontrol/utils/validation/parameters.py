#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:40:18 2026

Validation helpers for single input parameters

@author: johannes.wiesner
"""

import numpy as np

def _validate_choice(value, name, choices):
    """Validate an enumerated option."""
    if value not in choices:
        formatted_choices = ", ".join(
            repr(choice)
            for choice in choices
        )

        raise ValueError(
            f"{name} must be one of {formatted_choices}; "
            f"got {value!r}"
        )

def _validate_boolean(value, name):
    """Validate a boolean option."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a boolean"
        )

# TODO: Should be renamed _validate_positive_real_number
def _validate_positive_real(value, name):
    """Validate that input is a positive & finite real number."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        )
    ):
        raise TypeError(
            f"{name} must be a real number"
        )

    if not np.isfinite(value) or value <= 0:
        raise ValueError(
            f"{name} must be a positive finite number"
        )

def _validate_time_horizon(T, system):
    """Validate the control time horizon for the selected system."""
    if isinstance(T, (bool, np.bool_)) or not np.isscalar(T):
        raise TypeError(
            "T must be a scalar number"
        )

    if not np.isfinite(T) or T <= 0:
        raise ValueError(
            "T must be a positive finite number"
        )

    if system == "discrete":
        if not isinstance(T, (int, np.integer)) or T < 2:
            raise ValueError(
                "T must be an integer of at least 2 "
                "for a discrete system"
            )

    elif not isinstance(T, (float, np.floating)):
        raise TypeError(
            "T must be a float for a continuous system"
        )

def _validate_rho(rho, energy_type):
    """Validate rho for the selected energy type."""
    if energy_type == "minimal":
        if rho is not None:
            raise ValueError(
                "rho must be None when energy_type='minimal'."
            )

    elif energy_type == "optimal":
        if rho is None:
            raise ValueError(
                "rho must have a value when energy_type='optimal'."
            )

        if not isinstance(rho, (float, np.floating)):
            raise TypeError(
                "rho must be a float."
            )

        _validate_positive_real(rho, "rho")

        if rho > 1.0:
            raise ValueError(
                "rho must be less than or equal to 1."
            )