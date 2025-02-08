import numpy as np
from scipy.linalg import null_space

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