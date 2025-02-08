import numpy as np

def comm(x, y):
    return x @ y - y @ x


def components(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Compute the components of a matrix in a basis

    Args:
        matrix: the complex vector of shape (n,) to decompose
        basis: the basis of shape (k, n)
    """
    # linear regression almost works fine: minimize |c @ B - v|^2
    # except that the basis isn't full rank, so use lstsq
    return np.linalg.lstsq(basis.T, vector, 1e-8)[0]


def canonicalize(basis: np.ndarray) -> np.ndarray:
    """Change the scaling and phase of a basis so that it's unitary and 'nice'"""
    scale = (basis @ basis.conj().T)[0, 0]
    basis = basis / np.sqrt(scale)  # normalize properly
    first_nonzero = np.flatnonzero(np.abs(basis) > 1e-10)[0]
    phase = basis.ravel()[first_nonzero]
    phase /= abs(phase)
    basis /= phase
    return basis


def null_vec(mat: list[np.ndarray]) -> np.ndarray | None:
    epsilon = 1e-10  # singular values below this count as 0
    _, s, vh = np.linalg.svd(mat, full_matrices=False, compute_uv=True)
    if s[0] > epsilon:
        return None
    return vh[0].conj()