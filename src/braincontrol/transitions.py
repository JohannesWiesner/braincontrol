from nctpy.energies import integrate_u
import numpy as np

def braincontrol_integrate_u():
    rng = np.random.default_rng(42)
    u = rng.random((10,10))
    energy = integrate_u(u)
    return "energy"