#!python
# coding: utf-8
# @uthor: janak
"""CNN model based on Keras for *parameters guesses
    learned from POLLUX synthetic spectral database
    trained goto LAMOST release 1 spectral database
"""

import sys
import numpy as np
from keras.models import load_model
from tensorflow import device

from numpy import load
from itertools import cycle


def bachelor(file, base, batch=32):
    """Generate a batch of training pair of data from directory"""
    with device('/cpu:0'):
        names = open(file+".lst")
        datas = load(file+".npy")
        X = []; y = []; c = 0;
        
        while True:
            for name, data in cycle(zip(names, datas)):
                spec = load("{}/{}.npy".format(base, name.strip()))
                spec = spec.reshape(spec.shape + (1,))
                para = data.reshape((data.shape[0],))
                X.append(spec); y.append(para); c += 1
                if c == batch:
                    yield np.stack(X), np.stack(y)
                    X = []; y = []; c = 0;


if len(sys.argv) != 2:
    sys.exit("Pleace specify model name or dataset.")
else:
    name = sys.argv[1]

try:
    MODEL = load_model(name+".h5")
except (ImportError, ValueError) as _e_:
    sys.exit(_e_)
else:
    MODEL.summary()
    pass


with device('/gpu:0'):                                   # steps N*
    HISTORY = MODEL.fit_generator(bachelor("train", "linear", 32), 845, 50,
                                  validation_data=bachelor("tests", "linear", 32),
                                  validation_steps=282,
                                  verbose=1)

MODEL.save(name+".hdf5")
np.savetxt(name+".out", np.asarray([HISTORY.history["loss"],
                                    HISTORY.history["acc"],
                                    HISTORY.history["val_loss"],
                                    HISTORY.history["val_acc"]]).T)
