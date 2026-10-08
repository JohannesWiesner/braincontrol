#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:14:28 2026

@author: johannes.wiesner
"""

def _get_xr(xr, energy_type):
    """Resolve the reference state to the value required by nctpy.
    ncpty requires a non-None value for xr even when energy type is minimal"""
    
    # TODO: Would be nice if nctpy would also just accept None then we 
    # woulnd't have to do this
    if energy_type == "minimal":
        return "zero"

    return xr