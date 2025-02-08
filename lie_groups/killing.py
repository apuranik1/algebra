import numpy as np

from .types import MatrixLieAlgElt
from .representations import adjoint_matrices


def killing_form(basis: list[MatrixLieAlgElt]) -> np.ndarray:
    """Compute the Killing form of a Lie algebra given a basis"""
    adj_mats = adjoint_matrices(basis)
    flat = np.stack([mat.ravel() for mat in adj_mats], axis=1)
    flat_t = np.stack([mat.T.ravel() for mat in adj_mats], axis=0)
    return flat_t @ flat
