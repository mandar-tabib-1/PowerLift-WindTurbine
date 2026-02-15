"""
TT-OpInf: Simple Tensor Train with Operator Inference

A straightforward implementation for parametric spatio-temporal modeling
using only TT decomposition and OpInf (no GCA-ROM).
"""

from .tt_opinf import TT_OpInf

__all__ = ['TT_OpInf']
