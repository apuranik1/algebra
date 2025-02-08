import numpy as np
from scipy.linalg import null_space

from lie_groups.types import MatrixLieAlgElt
from lie_groups.util import components

def solve_equivalence(elts1, elts2):
    """Solves for an equivalence of representations.
    
    Specifically compute the set of matrices S such that SA = BS
    for each (A, B) in zip(elts1, elts2). Any such invertible matrix
    gives an isomorphism of the representations.
    """
    dim = len(elts1[0])
    blocks = []
    for a, b in zip(elts1, elts2):
        blocks.append(np.kron(np.eye(dim), a.T) - np.kron(b, np.eye(dim)))
    return null_space(np.concatenate(blocks, axis=0), rcond=1e-10).reshape(-1, dim, dim)

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