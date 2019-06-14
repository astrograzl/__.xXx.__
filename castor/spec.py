#!python
# coding: utf-8
# @uthor: janak
"""Reshape 1D spectrum to 2D square image with adding zeros"""
import numpy as np


def spec2sqr(spec, a=None, cut=False):
    """Reshape linear spectrum to square"""
    n = len(spec)
    assert n > 0
    if not a:
        if cut:
            a = int(np.floor(np.sqrt(n)))
        else:
            a = int(np.ceil(np.sqrt(n)))
    assert a > 0
    d = a**2 - n
    if d < 0:
        spec2d = spec[:d]
    else:
        spec2d = np.append(spec, np.zeros(a**2-n))
    return np.reshape(spec2d, (a, a))
