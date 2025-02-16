"""2D Euclidean Group"""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generic, Type, TypeVar

import numpy as np
import scipy.linalg as SLA
from numpy.typing import NDArray

from . import types


# class Action(metaclass=ABCMeta):
#     @abstractmethod
#     def group_on_group()

# def semidirect_product(group1: G1, alg1: A1, group2: G2, alg2: A2, action: Callable[[G1, A1], ]):
#     pass


def rotation(theta) -> NDArray[np.float64]:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


@dataclass(frozen=True)
class GroupElt(types.GroupElt):
    """Element of E(2).

    Any such element is a semidirect product of rotations and translations
    in 2D. The translation part is stored as a vector, and the rotations as
    an angle theta.
    """

    theta: float
    translation: NDArray[np.float64]

    def identity(cls: Type[Self]) -> Self:
        return cls(0.0, np.zeros(2))

    def rotation_matrix(self) -> np.ndarray:
        return rotation(self.theta)

    def __matmul__(self, other: Self) -> Self:
        rot = self.rotation
        return type(self)(
            self.theta + other.theta, self.translation + rot @ other.translation
        )

    def inverse(self) -> Self:
        rot = self.rotation_matrix()
        return type(self)(-self.theta, rot.T @ self.translation)

    def as_matrix(self: "GroupElt") -> np.ndarray:
        """Rep via matrices acting on homogeneous coordinates"""
        R = self.rotation_matrix()
        top = np.hstack([R, self.translation])
        return np.vstack([top, np.array([0, 0, 1])])

    @staticmethod
    def of_matrix(mat: NDArray[np.float64]) -> GroupElt:
        """Compute the group element corresponding to a matrix

        The matrix must act on homogeneous elements.
        """
        rotation = mat[:-1, :-1]
        translation = mat[:-1, -1]
        theta = np.arctan2(rotation[1, 0], rotation[0, 0])
        return GroupElt(theta, translation)


@dataclass(frozen=True)
class AlgElt(types.LieAlgElt[GroupElt]):
    """Element representing rotation at speed and translation at speed x"""

    theta: float
    offset: np.ndarray

    def __add__(self, other):
        return AlgElt(self.theta + other.theta, self.offset + other.offset)

    def __rmul__(self, x):
        return AlgElt(self.theta * x, self.offset * x)

    def as_matrix(self) -> np.ndarray:
        R = np.array([0, -self.theta], [self.theta, 0])
        top = np.hstack([R, self.offset])  # all but the last row
        return np.vstack(top, np.zeros(3))

    def bracket(self, other: Self) -> Self:
        theta = 0
        p1 = other.theta * self.offset[1] - self.theta * other.offset[1]
        p2 = self.theta * other.offset[0] - other.theta * self.offset[0]
        return type(self)(theta, np.array([p1, p2]))

    def conj_by(self, g: GroupElt) -> Self:
        # what does conjugating a group element look like?
        # conjugation by a translation feels like it should be trivial
        # (t, 1) (T, R) (-t, 1) = (t + T, R) (-t, 1) = (t + T - Rt, T)
        pass

    def exp(self) -> GroupElt:
        return GroupElt.of_matrix(SLA.expm(self.as_matrix()))
