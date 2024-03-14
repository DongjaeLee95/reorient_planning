#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass

@dataclass
class Param:
    dt: float
    Am: np.ndarray
    Am_inv: np.ndarray
    ur_max: np.ndarray
    ur_min: np.ndarray
    phidot_lb: float
    phidot_ub: float
    m: float
    g: float
    mu_dot: float
    L: float
    kf: float
    tilt_ang: float