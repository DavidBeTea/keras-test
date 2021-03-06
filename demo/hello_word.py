# -*- coding: utf-8 -*-
"""

@date 2018/6/24
@author: Zhao Xin
================

"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
import zoomutil.log as log

# Generate dummy data
x_train = np.random.random((10000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(10000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
for i in range(50):
    model.fit(x_train, y_train,
              epochs=20,
              batch_size=1024,verbose=0)
    score = model.evaluate(x_test, y_test, batch_size=128)
    log.info('now is %s/%s, score:%s' % (i, 5, score))
