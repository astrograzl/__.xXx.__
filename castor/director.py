#!python
# coding: utf-8
# @uthor: janak
"""Generate a pair of training or testing data from directory

>>> for x, y in director(data, base):
...     break

The input of this generator is name of file with list of inputs.
The output is a pair of two tensors. Which one contains the image
and second store the output vector. +1 #dimension
"""


from numpy import load
from tensorflow import device


def director(data, base):
    """Generate a pair of traning or testing data from directory"""
    with device('/cpu:0'):
        names = open(data+".lst")
        datas = load(data+".npy")

        while True:
            for i, spec in enumerate(names):
                dat = load(f"{base}/{spec.strip()}.npy")
                dat = dat.reshape((1,) + dat.shape + (1,))
                out = datas[i,:].reshape((1,) + (datas.shape[1],))
                yield dat, out
