import numpy as np

from .types import MatrixLieAlgElt
from .util import components

def adjoint_matrices(basis: list[MatrixLieAlgElt]) -> list[np.ndarray]:
    """Compute matrices of the adjoint representation given a basis of the algebra"""
    basis_vecs = np.stack([elt.matrix.ravel() for elt in basis])
    adjoint_matrices = []
    for element in basis:
        columns = []
        for basis_vec in basis:
            result = element.bracket(basis_vec)
            columns.append(components(result.matrix.ravel(), basis_vecs))
        adjoint_matrices.append(np.stack(columns, axis=1).real)
    return adjoint_matrices


def killing_form(basis: list[MatrixLieAlgElt]) -> np.ndarray:
    """Compute the Killing form of a Lie algebra given a basis"""
    adj_mats = adjoint_matrices(basis)
    flat = np.stack([mat.ravel() for mat in adj_mats], axis=1)
    flat_t = np.stack([mat.T.ravel() for mat in adj_mats], axis=0)
    return flat_t @ flat
