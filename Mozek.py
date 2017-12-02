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


try:
    MODEL = load_model("pollux.h5")
except (ImportError, ValueError) as _e_:
    sys.exit(_e_)
else:
    MODEL.summary()


with device('/gpu:0'):
    HISTORY = MODEL.fit_generator(director("train"), 256, 1000,
                                  validation_data=director("test"),
                                  validation_steps=96,
                                  verbose=1)

MODEL.save("pollux.hdf5")
np.savetxt("pollux.out", np.asarray([HISTORY.history["loss"],
                                     HISTORY.history["acc"],
                                     HISTORY.history["val_loss"],
                                     HISTORY.history["val_acc"]]).T)
