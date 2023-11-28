import numpy as np

from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Dense,  BatchNormalization, Dropout, LeakyReLU, Flatten
from tensorflow.keras import Input
from tensorflow.keras import Model
from keras.utils.vis_utils import plot_model

import config


def conv_network():
    # Network parameters
    conv_filters = 64 # 32
    kernel_size = (3, 3)
    pool_size = 2
    act = None  # 'relu'
    input_data = Input(name='the_input', shape=(config.img_width, config.img_height, 1), dtype='float32')
    inner = input_data

    for i in range(int(np.ceil(np.log2(min(config.img_width, config.img_height))))):
        inner = Conv2D(conv_filters, kernel_size, padding='same', activation=act)(inner)
        inner = Conv2D(conv_filters, kernel_size, padding='same', activation=act)(inner)
        inner = Conv2D(conv_filters, kernel_size, padding='same', activation=act)(inner)
        inner = LeakyReLU(0.1)(inner)
        inner = Dropout(0.2)(inner)
        inner = BatchNormalization()(inner)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size))(inner)

    inner = Flatten()(inner)
    inner = Dense(1024)(inner)
    inner = Dense(512)(inner)
    inner = Dense(512)(inner)
    # inner = Dense(len(config.categories.keys()), activation='softmax')(inner)
    inner = Dense(config.output_size, activation='softmax')(inner)

    model = Model(inputs=input_data, outputs=inner)

    return model


def create_conv2D():
    """Creates CNN model"""
    filters = 32
    input_nn = Input(shape=(config.img_width, config.img_height, config.img_channels))
    x = Conv2D(filters=filters, kernel_size=(2, 4))(input_nn)
    x = Conv2D(filters=filters, kernel_size=(2, 4))(x)
    x = Conv2D(filters=filters, kernel_size=(2, 4))(x)

    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 4), 3)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(filters=filters, kernel_size=(2, 3))(x)
    x = Conv2D(filters=filters, kernel_size=(2, 3))(x)
    x = Conv2D(filters=filters, kernel_size=(2, 3))(x)
    x = LeakyReLU(0.1)(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(256, activation="relu")(x)

    x = Dense(config.output_size, activation="softmax")(x)

    model = Model(inputs=input_nn, outputs=x)

    return model


