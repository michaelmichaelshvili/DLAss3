import os

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import LSTM, Conv2D, MaxPool2D, Flatten, Dense, Input, Subtract, Lambda
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
import keras.backend as K

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from DataPreparation import *

window_size = 5

product_title_input, search_term_input = load_product_title_search_term_lists(window_size)

model = Sequential()
model.add(LSTM(1,activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mse')

