# -*- coding: utf-8 -*-
"""

@date 2018/6/24
@author: Zhao Xin
================

"""
import numpy as np
import pandas as pd
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

model = Sequential()
model.add(Embedding(input_dim=1024, output_dim=256, input_length=50))
model.add(LSTM(128))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

plot_model(model, to_file='model1.png', show_shapes=True)
