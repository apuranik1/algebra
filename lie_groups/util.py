import numpy as np

def comm(x, y):
    return x @ y - y @ x


def components(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Compute the components of a matrix in a basis

    Args:
        matrix: the complex vector of shape (n,) to decompose
        basis: the basis of shape (k, n)
    """
    # solve almost works fine: minimize |c @ B - v|^2
    # except that the basis isn't full rank, so use lstsq
    return np.linalg.lstsq(basis.T, vector, 1e-8)[0]


def canonicalize(basis: np.ndarray) -> np.ndarray:
    """Change the scaling and phase of a basis so that it's unit norm and 'nice'"""
    scale = (basis @ basis.conj().T)[0, 0]
    basis = basis / np.sqrt(scale)  # normalize properly
    first_nonzero = np.flatnonzero(np.abs(basis) > 1e-10)[0]
    phase = basis.ravel()[first_nonzero]
    phase /= abs(phase)
    basis /= phase
    return basis


def orth_complement(basis: np.ndarray, subspace_basis: np.ndarray) -> np.ndarray:
    """Return a basis of the orthogonal complement of a subspace.
    The full space is the column span of `basis`, and the subspace is the column
    span of `subspace_basis`.
    """
    frob = (basis ** 2).sum() ** 0.5
    proj_coefs = np.linalg.lstsq(subspace_basis, basis, rcond=1e-8)[0]
    proj = subspace_basis @ proj_coefs
    remainder = basis - proj
    u, s, _ = np.linalg.svd(remainder)
    n_nonzero = (s > frob * 1e-8).sum()
    return u[:, :n_nonzero]
