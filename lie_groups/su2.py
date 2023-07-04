from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import numpy.linalg as LA
import scipy.linalg as SLA


@dataclass
class Group:
    matrix: np.ndarray

    def __post_init__(self):
        self.assert_valid()

    def assert_valid(self):
        assert self.matrix.shape == (2, 2)
        assert np.allclose(LA.det(self.matrix), 1)
        assert np.allclose(self.matrix @ self.matrix.conj().T, np.eye(2))

    @staticmethod
    def sample(rand: Optional[np.random.Generator] = None) -> Group:
        if rand is None:
            rand = np.random.default_rng()
        # sample from 3-sphere
        vec = rand.standard_normal(4)
        vec /= LA.norm(vec)
        ar, ai, br, bi = vec
        a = ar + ai * 1j
        b = br + bi * 1j
        matrix = np.array([[ar + ai * 1j, br + bi * 1j], [-br + bi * 1j, ar - ai * 1j]])
        return Group(matrix)

    def inverse(self) -> Group:
        return Group(self.matrix.conj().T)

    def __mul__(self, other: Group) -> Group:
        return Group(self.matrix @ other.matrix)


@dataclass
class LieAlg:
    matrix: np.ndarray

    def __post_init__(self):
        self.assert_valid()

    def assert_valid(self):
        # skew-symmetric and traceless
        assert np.allclose(self.matrix, -self.matrix.conj().T)
        assert np.allclose(self.matrix.trace(), 0.)

    def exp(self) -> Group:
        return Group(SLA.expm(self.matrix))

    def conj_by(self, g: Group):
        return g.matrix @ self.matrix @ g.inverse().matrix


def group_rep(dimension: int) -> Callable[[Group], np.ndarray]:
    """Realize the unique irrep of a given dimension.
    
    In 1D, produces the trivial rep; in 2D, the defining rep.
    """
    assert dimension >= 1
    def rep(elt: Group) -> np.ndarray:
        deg = dimension - 1  # degree of polynomials
        ft = np.fft.fft(elt.matrix, dimension)
        w1, w2 = ft
        rows = []
        for i in range(dimension):
            rows.append(np.fft.ifft(w1 ** (deg - i) * w2 ** i))
        return np.stack(rows)

    return rep