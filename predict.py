import cv2
import numpy as np
from keras_adabound import AdaBound
from tensorflow.keras.models import load_model


def make_cutout2(img, bbox):
    w = bbox[2] - bbox[0]
    w_c = 1.5
    bbox[0] = max(0, bbox[0] - int(w_c * (1.5 * w)))
    bbox[2] = min(bbox[2] + int(w_c * (1.5 * w)), img.shape[1])
    bbox[1] = max(0, bbox[1] - 3 * w)
    bbox[3] = min(bbox[3] + w, img.shape[0])
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]


class ColourClassification:
    def __init__(self, model_path: str):
        self.model = load_model(model_path, custom_objects={'AdaBound': AdaBound})
        self.img_width = 128
        self.img_height = 128
        self.img_channels = 3
        self.output_size = 9

        self.classes_dict = {
            'black': 0,
            'silver-grey': 1,
            'white': 2,
            'red': 3,
            'blue': 4,
            'brown': 5,
            'green': 6,
            'yellow': 7,
            'orange': 8,
        }

    def predict_on_image(self, img_cut):
        img_res = cv2.resize(img_cut, (self.img_width, self.img_height))

        data = np.array((img_res - np.min(img_res)) / (np.max(img_res) - np.min(img_res)), dtype=np.float32)
        if len(data.shape) < 4:
            data = np.reshape(data, ((1,) + data.shape))

        result = self.model.predict(data)
        output = np.argmax(result)
        predicted_label = list(self.classes_dict.keys())[list(self.classes_dict.values()).index(output)]

        return predicted_label


if __name__ == '__main__':
    cc = ColourClassification('CC_bestNetwork.hdf5')
