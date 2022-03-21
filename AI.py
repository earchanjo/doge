import cv2 as cv2
import numpy as np
import PIL
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard

img_rows, img_cols = 40,40
def buildmodel():

    model = Sequential()
    model.add(Conv2D(32, (8,8), strides=(4,4), padding='same', input_shape=(img_cols, img_rows, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4,4), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3), strides=(1,1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))

    adam = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=adam)