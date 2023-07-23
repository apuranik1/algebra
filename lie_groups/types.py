from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Self, TypeVar, Type

import numpy as np
from numpy.typing import NDArray
import numpy.linalg as LA
import scipy.linalg as SLA


G = TypeVar("G", bound="GroupElt")


class GroupElt:
    """A group. `G` should be the same as the class."""

    @classmethod
    @abstractmethod
    def identity(cls: Type[Self]) -> Self:
        pass

    @abstractmethod
    def __matmul__(self, other: Self) -> Self:
        pass

    @abstractmethod
    def inverse(self) -> Self:
        pass


class LieAlgElt(Generic[G]):
    """Lie algebra over the group `G`.

    `A` should be the same as the class.
    """

    @abstractmethod
    def bracket(self, other: Self) -> Self:
        """Lie bracket.

        `*` is reserved for scalar multiplication, and `@` is reserved for the
        product in the universal enveloping algebra.
        """
        pass

    @abstractmethod
    def conj_by(self, g: G) -> Self:
        """Compute the conjugation g * self * g^{-1}"""
        pass

    @abstractmethod
    def __add__(self, other: Self) -> Self:
        pass

    @abstractmethod
    def __rmul__(self, scalar: complex) -> Self:
        """Multiply by a scalar.

        Really this should only allow real numbers, but we pretty much always
        care about the complexification.
        """
        pass


@dataclass(frozen=True)
class MatrixGroupElt(GroupElt):
    """Matrix group. Not always the most efficient representation."""

    matrix: NDArray[np.complex_]

    @classmethod
    @abstractmethod
    def dimension(cls) -> int:
        pass

    @classmethod
    def identity(cls) -> Self:
        return cls(np.eye(2, dtype=np.complex_))

    def __matmul__(self: Self, other: Self) -> Self:
        return type(self)(self.matrix @ other.matrix)

    def inverse(self) -> Self:
        return type(self)(LA.inv(self.matrix))


MG = TypeVar("MG", bound=MatrixGroupElt)


@dataclass(frozen=True)
class MatrixLieAlgElt(LieAlgElt[MG]):
    """Lie algebra of a matrix group

    Also implements the universal enveloping algebra.
    Subclasses should validate in __post_init__.
    """

    matrix: NDArray[np.complex_]

    @classmethod
    @abstractmethod
    def group(cls) -> Type[MG]:
        pass

    @classmethod
    @abstractmethod
    def basis(cls) -> list[Self]:
        pass

    def __add__(self, other: Self) -> Self:
        return type(self)(self.matrix + other.matrix)

    def __rmul__(self, scalar: complex) -> Self:
        return type(self)(scalar * self.matrix)

    def __matmul__(self, other: Self) -> Self:
        """Matrix product in the universal enveloping algebra."""
        return type(self)(self.matrix @ other.matrix)

    def bracket(self: Self, other: Self) -> Self:
        return type(self)(self.matrix @ other.matrix - other.matrix @ self.matrix)

    def exp(self) -> MG:
        return self.group()(SLA.expm(self.matrix))  # type: ignore

    def conj_by(self, g: MG) -> Self:
        return type(self)(g.matrix @ self.matrix @ g.inverse().matrix)
