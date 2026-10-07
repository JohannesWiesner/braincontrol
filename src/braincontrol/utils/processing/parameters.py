#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:24:44 2026

Helpers to resolve single value inputs

@author: johannes.wiesner
"""

def _get_rho(rho, energy_type):
    """Resolve rho to the value required by nctpy."""
    if energy_type == "minimal":
        # nctpy requires positive rho even when S is zero.
        return 1.0

    return rho