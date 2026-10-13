#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:14:28 2026

@author: johannes.wiesner
"""

def _get_xr(xr, energy_type):
    """Resolve symbolic or omitted references after validation.

    nctpy 1.2.0 requires a non-None reference even for minimal energy
    (a zero state-cost matrix), so the public xr=None resolves to "zero".
    Array shapes are normalized by Transitioner._process_states.
    """
    
    # TODO: Would be nice if nctpy would also just accept None then we 
    # woulnd't have to do this
    if energy_type == "minimal":
        return "zero"

    return xr