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
from castor import director


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
    HISTORY = MODEL.fit_generator(director("train", name), 512, 100,
                                  validation_data=director("tests", name),
                                  validation_steps=128,
                                  verbose=2)

MODEL.save(name+".hdf5")
np.savetxt(name+".out", np.asarray([HISTORY.history["loss"],
                                    HISTORY.history["acc"],
                                    HISTORY.history["val_loss"],
                                    HISTORY.history["val_acc"]]).T)
