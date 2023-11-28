from tensorflow.keras.layers import concatenate, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization, Dropout, GlobalAveragePooling2D, LeakyReLU, Flatten
from tensorflow.python.keras.layers import Lambda
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import load_model
from keras_adabound import AdaBound
import config

def create_branch(img_input, branch_name):
    #######  BLOCK1 #################
    b1_filters = 48
    b1_kernel = 11
    b1_stride = 4
    b1_padding = 'valid'
    b1_kernel_init = 'he_normal'

    block_name = f'{branch_name}_block1'
    x1 = Conv2D(filters=b1_filters, kernel_size=b1_kernel, strides=b1_stride, padding=b1_padding, kernel_initializer=b1_kernel_init, name=f'{block_name}_conv')(img_input)
    x1 = BatchNormalization(name=f'{block_name}_bn')(x1)
    x1 = Activation('relu', name=f'{block_name}_relu')(x1)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=f'{block_name}_pooling')(x1)

    #######  BLOCK2 SPLIT #################
    b2_filters = 64
    b2_kernel = 3
    b2_stride = 1
    b2_padding = 'same'
    b2_kernel_init = 'he_normal'

    block_name = f'{branch_name}_block2'
    channels = K.int_shape(x1)[-1]
    top_branch_2 = Lambda(lambda x: x[:, :, :, :channels // 2], name=f'{block_name}_split1')(x1)
    bottom_branch_2 = Lambda(lambda x: x[:, :, :, channels // 2:], name=f'{block_name}_split2')(x1)
    split_layers_2 = [top_branch_2, bottom_branch_2]

    for i in range(2):
        split_layers_2[i] = Conv2D(filters=b2_filters, kernel_size=b2_kernel, strides=b2_stride, padding=b2_padding, kernel_initializer=b2_kernel_init, name=f'{block_name}_conv_{i}')(split_layers_2[i])
        split_layers_2[i] = BatchNormalization(name=f'{block_name}_Bn_{i}')(split_layers_2[i])
        split_layers_2[i] = Activation('relu', name=f'{block_name}_relu_{i}')(split_layers_2[i])
        split_layers_2[i] = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=f'{block_name}_pooling_{i}')(split_layers_2[i])

    x1_top = split_layers_2[0]
    x1_bot = split_layers_2[1]

    #######  BLOCK3  #################
    b3_filters = 96
    b3_kernel = 3
    b3_stride = 1
    b3_padding = 'same'
    b3_kernel_init = 'he_normal'

    block_name = f'{branch_name}_block3t'
    x1_top = Conv2D(filters=b3_filters, kernel_size=b3_kernel, strides=b3_stride, padding=b3_padding, kernel_initializer=b3_kernel_init, name=f'{block_name}_branch_conv')(x1_top)
    x1_top = BatchNormalization(name=f'{block_name}_branch_bn')(x1_top)
    x1_top = Activation('relu', name=f'{block_name}_branch_relu')(x1_top)

    block_name = f'{branch_name}_block3b'
    x1_bot = Conv2D(filters=b3_filters, kernel_size=b3_kernel, strides=b3_stride, padding=b3_padding, kernel_initializer=b3_kernel_init, name=f'{block_name}_branch_conv')(x1_bot)
    x1_bot = BatchNormalization(name=f'{block_name}_branch_bn')(x1_bot)
    x1_bot = Activation('relu', name=f'{block_name}_branch_relu')(x1_bot)

    x1 = concatenate([x1_top, x1_bot], axis=-1, name=f'{branch_name}_block3_concatenate')

    #######  BLOCK4 SPLIT  #################
    block_name = f'{branch_name}_block4'
    b4_filters = 96
    b4_kernel = 3
    b4_stride = 1
    b4_padding = 'same'
    b4_kernel_init = 'he_normal'

    channels = K.int_shape(x1)[-1]
    top_branch_4 = Lambda(lambda x: x[:, :, :, :channels // 2], name=f'{block_name}_split1')(x1)
    bottom_branch_4 = Lambda(lambda x: x[:, :, :, channels // 2:], name=f'{block_name}_split2')(x1)
    split_layers_4 = [top_branch_4, bottom_branch_4]

    for i in range(2):
        split_layers_4[i] = Conv2D(filters=b4_filters, kernel_size=b4_kernel, strides=b4_stride, padding=b4_padding, kernel_initializer=b4_kernel_init, name=f'{block_name}_conv_{i}')(split_layers_4[i])
        # split_layers_4[i] = BatchNormalization(name=f'{block_name}_Bn_{i}')(split_layers_4[i])
        split_layers_4[i] = Activation('relu', name=f'{block_name}_relu_{i}')(split_layers_4[i])

    x1_top = split_layers_4[0]
    x1_bot = split_layers_4[1]

    #######  BLOCK5   #################
    b5_filters = 64
    b5_kernel = 3
    b5_stride = 1
    b5_padding = 'same'
    b5_kernel_init = 'he_normal'

    block_name = f'{branch_name}_block5t'
    x1_top = Conv2D(filters=b5_filters, kernel_size=b5_kernel, strides=b5_stride, padding=b5_padding, kernel_initializer=b5_kernel_init, name=f'{block_name}_branch_conv')(x1_top)
    x1_top = BatchNormalization(name=f'{block_name}_branch_bn')(x1_top)
    x1_top = Activation('relu', name=f'{block_name}_branch_relu')(x1_top)
    x1_top = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=f'{block_name}_pooling')(x1_top)

    block_name = f'{branch_name}_block5b'
    x1_bot = Conv2D(filters=b5_filters, kernel_size=b5_kernel, strides=b5_stride, padding=b5_padding, kernel_initializer=b5_kernel_init, name=f'{block_name}_branch_conv')(x1_bot)
    x1_bot = BatchNormalization(name=f'{block_name}_branch_bn')(x1_bot)
    x1_bot = Activation('relu', name=f'{block_name}_branch_relu')(x1_bot)
    x1_bot = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name=f'{block_name}_pooling')(x1_bot)

    return x1_top, x1_bot


def create_conv2D_multichannel():
    """Creates CNN model"""
    img_input = Input(shape=(config.img_width, config.img_height, config.img_channels))
    ## BRANCHES
    x1_top, x1_bot = create_branch(img_input, branch_name='TOP')
    x2_top, x2_bot = create_branch(img_input, branch_name='BOT')

    ## CONCATE
    img_output = concatenate([x1_top, x1_bot, x2_top, x2_bot], axis=-1, name='features_output')

    if 1:
        img_output = GlobalAveragePooling2D(name='GAP_layer')(img_output)
        img_output = Dense(config.output_size, activation='softmax', kernel_initializer='he_normal', name='color_output')(img_output)
    else:
        img_output = Flatten(name='flatten')(img_output)
        img_output = Dense(4096, activation='softmax', kernel_initializer='he_normal', name='fc1')(img_output)
        img_output = Dense(1024, activation='softmax', kernel_initializer='he_normal', name='fc2')(img_output)
        img_output = Dense(256, activation='softmax', kernel_initializer='he_normal', name='fc3')(img_output)
        img_output = Dense(config.output_size, activation='softmax', kernel_initializer='he_normal', name='color_output')(img_output)

    model = Model(inputs=img_input, outputs=img_output)

    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model


