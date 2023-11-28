import config
import os
import random
import numpy as np
import cv2


def create_dataset(dataset_path, classes_dict, normalize=False):
    file_labels = {}
    files = []
    with open(dataset_path, 'r') as fd:
        for line in fd.readlines():
            splitted = line[:-1].split(';')
            _, f_name = os.path.split(splitted[0])
            colour = splitted[1]
            file_labels[f_name] = colour
            files.append(splitted[0])

    random.shuffle(files)
    train_list = files[0:int((1 - config.validation_split) * len(files))]
    val_list = files[int((1 - config.validation_split) * len(files)):len(files)]

    print("Total samples: {}".format(len(files)))
    print("Training set size: {}".format(len(train_list)))
    print("Validation set size: {}".format(len(val_list)))

    if normalize:
        train_data = np.zeros((len(train_list), config.img_height, config.img_width, config.img_channels), dtype=np.float32)
        val_data = np.zeros((len(val_list), config.img_height, config.img_width, config.img_channels), dtype=np.float32)
    else:
        # train_data = np.zeros((len(train_list), config.img_height, config.img_width, config.img_channels), dtype=np.uint8)
        # val_data = np.zeros((len(val_list), config.img_height, config.img_width, config.img_channels), dtype=np.uint8)

        train_data = []
        val_data = []

    train_labels = []
    val_labels = []

    print("Creating training dataset")

    for i, file in enumerate(train_list):
        fol_name, fil_name = os.path.split(file)

        if fil_name in file_labels:
            colour = file_labels[fil_name]
            class_id = int(classes_dict[colour])

        else:
            continue

        try:
            img = cv2.imread(file)
            img_res = cv2.resize(img, (config.img_width, config.img_height))
            if normalize:
                img_norm = cv2.normalize(img_res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                train_data.append(img_norm)
            else:
                train_data.append(img_res)

            label = [0]*config.output_size
            label[class_id] = 1
            train_labels.append(label)

        except Exception as e:
            print(e)
            continue

        if i % 1000 == 0 and i > 0: print(".. training samples processed {}".format(i))

    train_labels = np.array(train_labels).reshape((len(train_labels), config.output_size))
    np.save(config.train_data_npy, train_data)
    np.save(config.train_labels_npy, train_labels)

    print("Creating validation dataset")

    for i, file in enumerate(val_list):
        fol_name, fil_name = os.path.split(file)

        if fil_name in file_labels:
            colour = file_labels[fil_name]
            class_id = int(classes_dict[colour])

        else:
            continue

        try:
            img = cv2.imread(file)
            img_res = cv2.resize(img, (config.img_width, config.img_height))
            if normalize:
                img_norm = cv2.normalize(img_res, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                val_data.append(img_norm)
            else:
                val_data.append(img_res)

            label = [0] * config.output_size
            label[class_id] = 1
            val_labels.append(label)

        except Exception as e:
            print(e)
            continue

        if i % 1000 == 0 and i > 0: print(".. validation samples processed {}".format(i))

    val_labels = np.array(val_labels).reshape((len(val_labels), config.output_size))
    np.save(config.val_data_npy, val_data)
    np.save(config.val_labels_npy, val_labels)


if __name__ == '__main__':
    create_dataset(config.labels_path, config.classes_dict, normalize=False)

