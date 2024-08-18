import numpy as np
from scipy.linalg import block_diag

from . import so3

def make_boost(i):
    arr = np.zeros((4, 4))
    arr[i, 0] = arr[0, i] = 1
    return arr

J1 = block_diag([[0]], so3.L1)
J2 = block_diag([[0]], so3.L2)
J3 = block_diag([[0]], so3.L3)
K1 = make_boost(1)
K2 = make_boost(2)
K3 = make_boost(3)

_J = [J1, J2, J3]
_K = [K1, K2, K3]

def J(i):
    return _J[i - 1]

def K(i):
    return _K[i - 1]

def Jp(i):
    return 0.5 * (J(i) + 1j * K(i))

def Jm(i):
    return 0.5 * (J(i) - 1j * K(i))