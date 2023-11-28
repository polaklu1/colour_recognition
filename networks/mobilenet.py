import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model

# from .. import config
import config


def create_mobilenet():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(config.img_width, config.img_height, config.img_channels), alpha=.5,
                                                   include_top=False,
                                                   weights='imagenet')

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Dropout(.5)(x)
    prediction_layer = tf.keras.layers.Dense(config.output_size, activation='softmax')(x)

    learning_rate = 0.00001

    model = Model(inputs=base_model.input, outputs=prediction_layer)

    for layer in model.layers[:80]:
        layer.trainable=False
    for layer in model.layers[80:]:
        layer.trainable=True

    return model






