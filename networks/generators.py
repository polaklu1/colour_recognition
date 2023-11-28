import config
import numpy as np
from .image_augmentation import augment_image


def random_number_from_range(f, t):
    while True:
        random_order = np.arange(f, t, 1)
        np.random.shuffle(random_order)
        for i in range(len(random_order)):
            yield random_order[i]


def prepare_classes(data):
    # data = [img, label]
    classes = [[None]*3] * config.output_size
    for d in range(len(data[0])):
        c = np.argmax(data[1][d])

        if classes[c][1] is None:
            classes[c] = [None, [data[0][d]], data[1][d] ]
        else:
            classes[c][1].append(data[0][d])

    for i in range(config.output_size):
        if classes[i][1] is not None:
            classes[i][0] = random_number_from_range(0, len(classes[i][1]))
        else:
            classes[i] = None
    return classes


def generator_with_noise_equal_classes(source, batch_size, aug_pipeline=None):
    next_id = random_number_from_range(0, config.output_size)

    classes = prepare_classes(source)

    while True:

        data = np.zeros((batch_size, config.img_width, config.img_height, config.img_channels), dtype=np.uint8)

        labels = np.zeros((batch_size, config.output_size))
        for i in range(batch_size):
            class_id = next(next_id)
            while classes[class_id] is None:
                    class_id = next(next_id)

            in_class_id = next(classes[class_id][0])
            data[i] = classes[class_id][1][in_class_id]
            labels[i] = classes[class_id][2]

        if aug_pipeline:
            data = augment_image(data, aug_pipeline)
        else:
            data = np.array((data - np.min(data)) / (np.max(data) - np.min(data)), dtype=np.float32)

        yield data, labels
