import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config_tf)
set_session(sess)

########## CPU
# tf.config.set_visible_devices([], 'GPU')
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#######################################
import config
import time
import cv2
import os
import numpy as np

from networks.conv2D_multichannel import create_conv2D_multichannel
from networks.generators import generator_with_noise_equal_classes
from networks.image_augmentation import build_pipeline
from tensorflow.keras.models import load_model
from utils import get_current_timestamp


def show_img(data, label=None):
    """ Converts image to uint8 and shows image"""
    data = (data - data.min()) / (data.max() - data.min()) * 255
    data = data.astype(np.uint8)

    if data.shape[2] == 2:
        img_zero = np.zeros((data.shape[0], data.shape[1], 1), dtype=np.uint8)
        img_show_stack = np.dstack((data, img_zero))
        cv2.imshow("Img", cv2.resize(img_show_stack, (600, 384)))
        cv2.waitKey(0)


def test_network(model):
    test_data = np.load(config.val_data_npy)
    test_labels = np.load(config.val_labels_npy)

    print("Test data loaded")
    print("Evaluation on dataset")

    prediction_time = 0

    cntr = 0
    wrong = [0] * config.output_size
    correct = [0] * config.output_size
    second_guess = np.zeros((config.output_size, config.output_size))
    guesses = np.zeros((config.output_size, config.output_size))

    for i in range(0, len(test_data), config.validation_batch_size):
        data = test_data[i:i + config.validation_batch_size]
        data = np.array((data - np.min(data)) / (np.max(data) - np.min(data)), dtype=np.float32)
        labels = test_labels[i:i + config.validation_batch_size]

        t_start = time.clock()
        result = model.predict(data)
        prediction_time += time.clock() - t_start

        for i in range(len(result)):
            output = np.argmax(result[i])
            label = np.argmax(labels[i])
            guesses[label] += result[i]

            if output == label:
                cntr += 1
                correct[label] += 1

            else:
                wrong[label] += 1
                second_guess[label][output] += 1

    print('Average prediction time: {}'.format(prediction_time / len(test_data)))

    category_id = config.classes_dict

    lines_to_export = []
    with open('log.txt', 'w+') as fd:
        for i in range(config.output_size):
            if correct[i] + wrong[i] > 0:

                car_cat = list(category_id.keys())[list(category_id.values()).index(i)]
                second_guess[i][i] = 0
                second_cat = list(category_id.keys())[list(category_id.values()).index(np.argmax(second_guess[i]))]
                second_cat_val = np.max(second_guess[i])
                cat_count = correct[i] + wrong[i]

                print("{0} - correct: {1} x wrong {2} - accuracy: {3} - second_guess: {4} - with {5}".format(car_cat, correct[i], wrong[i], (float(correct[i]) / cat_count if cat_count != 0 else 1), second_cat, second_cat_val))
                fd.write("{0} - correct: {1} x wrong {2} - accuracy: {3} - second_guess: {4} - with {5} \n".format(car_cat, correct[i], wrong[i], (float(correct[i]) / cat_count if cat_count != 0 else 1), second_cat, second_cat_val))
                line = "{0};{1};{2};{3};{4};{5} \n".format(car_cat, correct[i], wrong[i], (float(correct[i]) / cat_count if cat_count != 0 else 1), second_cat, second_cat_val)
                lines_to_export.append(line)
        np.set_printoptions(linewidth=np.inf)

        for i in range(config.output_size):
            print(guesses[i])

        print("total: {0}".format(float(cntr)/len(test_labels)))
        fd.write("total: {0}\n".format(float(cntr)/len(test_labels)))
        fd.close()

    with open('colour_recognition_statistic_{}.csv'.format(get_current_timestamp()), 'w+') as fd:
        fd.write('model;correct;wrong;accuracy;second guess;second guess count\n')
        for l in lines_to_export:
            fd.write(l)


def predict_on_data(model, data):
    data = np.array((data - np.min(data)) / (np.max(data) - np.min(data)), dtype=np.float32)
    result = model.predict(data)
    return result


def compile_model(model):
    model.compile(config.optimizer, config.loss, config.metrics)
    model.summary()
    print("network compiled")
    return model


def train_network_with_generator(model):
    """Model training on dataset with generator """
    train_data = np.load(config.train_data_npy)
    train_labels = np.load(config.train_labels_npy)
    val_data = np.load(config.val_data_npy)
    val_labels = np.load(config.val_labels_npy)

    number_of_train_samples = len(train_data)
    number_of_validation_samples = len(val_data)
    steps_per_epoch = int(len(train_data) / float(config.batch_size)) + 1
    validation_steps = int(len(val_data) / float(config.batch_size)) + 1
    print("Number of training samples: {}".format(number_of_train_samples))
    print("Number of validation samples: {}".format(number_of_validation_samples))

    aug_pipeline = build_pipeline()
    train_generator = generator_with_noise_equal_classes([train_data, train_labels], config.batch_size, aug_pipeline=aug_pipeline)
    validation_generator = generator_with_noise_equal_classes([val_data, val_labels], config.validation_batch_size, aug_pipeline=None)

    print("Starting model training")

    if os.name == 'nt':
        print("Running win configuration")
        model.fit_generator(train_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=validation_generator,
                            validation_steps=validation_steps,
                            epochs=config.epochs, verbose=1,
                            callbacks=[config.save_model_callback])
    else:
        print("Running unix configuration")
        model.fit_generator(train_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=validation_generator,
                            validation_steps=validation_steps,
                            epochs=config.epochs, verbose=1,
                            callbacks=[config.save_model_callback],
                            # use_multiprocessing=True,
                            workers=2)

    return model


if __name__ == '__main__':
    if 1:
        model = create_conv2D_multichannel()

        model = compile_model(model)
        model = train_network_with_generator(model)
    else:
        # model = load_model(config.best_network)
        from keras_adabound import AdaBound
        model = load_model(config.best_network, custom_objects={'AdaBound': AdaBound})
        testNetwork(model)

