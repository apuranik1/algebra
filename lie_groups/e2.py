"""2D Euclidean Group"""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
import numpy.linalg as LA
import scipy.linalg as SLA

from . import types


# class Action(metaclass=ABCMeta):
#     @abstractmethod
#     def group_on_group()

# def semidirect_product(group1: G1, alg1: A1, group2: G2, alg2: A2, action: Callable[[G1, A1], ]):
#     pass


@dataclass(frozen=True)
class GroupElt:
    """Element of E(2).
    
    Any such element is a semidirect product of rotations and translations
    in 2D. The translation part is stored as a vector, and the rotations as
    an angle theta.
    """
    theta: float
    translation: npt.NDArray[np.float64]