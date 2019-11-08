"""
Single-Particle 1D Quantum Mechanics module.
"""
from .constants import *
try:
    # raise ImportError
    from .qm_numba import *
except ImportError:
    from .qm import *
