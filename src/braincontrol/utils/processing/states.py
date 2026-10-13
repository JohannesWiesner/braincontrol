#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:14:28 2026

@author: johannes.wiesner
"""

def _get_xr(xr, energy_type):
    """Resolve the reference state to the value required by nctpy.

    nctpy 1.2.0 requires a non-None reference even when S is zero for
    minimal energy.
    """
    if energy_type == "minimal":
        return "zero"

    return xr
