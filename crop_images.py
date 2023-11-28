import json
import cv2
import os
import numpy as np

# from scipy.spatial import distance as dist
#
#
# def order_points(pts):
#     xSorted = pts[np.argsort(pts[:, 0]), :]
#     leftMost = xSorted[:2, :]
#     rightMost = xSorted[2:, :]
#
#     leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
#     (tl, bl) = leftMost
#
#     D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
#     (br, tr) = rightMost[np.argsort(D)[::-1], :]
#
#     return np.array([tl, tr, br, bl], dtype="int32")


EXPORT_PATH = 'EXPORT_CROPPED'

def make_cutout(img, bbox):
    w = bbox[2] - bbox[0]
    w_c = 1.5
    bbox[0] = max(0, bbox[0] - int(w_c * w))
    bbox[2] = min(bbox[2] + int(w_c * w), img.shape[1])
    bbox[1] = max(0, bbox[1] - 2 * w)
    bbox[3] = min(bbox[3] + w, img.shape[0])
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]


if __name__ == '__main__':
    os.makedirs(EXPORT_PATH, exist_ok=True)

    with open('files_labeled_lp_colour_path_2023_02_10T17_16.json', 'r') as fd:
        labels = json.load(fd)

    for fil in labels:
        lab = labels[fil]
        fil_name = os.path.split(fil)[-1]
        try:
            img = cv2.imread(fil)
            # X1;Y1;X2;Y2;X3;Y3;X4;Y4;OrigLine
            height, width = img.shape[:2]

            x1 = (float(lab['label'][0]) * width)
            y1 = (float(lab['label'][1]) * height)
            x2 = (float(lab['label'][2]) * width)
            y2 = (float(lab['label'][3]) * height)
            x3 = (float(lab['label'][4]) * width)
            y3 = (float(lab['label'][5]) * height)
            x4 = (float(lab['label'][6]) * width)
            y4 = (float(lab['label'][7]) * height)

            p1 = tuple(map(int, [x1, y1]))
            p2 = tuple(map(int, [x2, y2]))
            p3 = tuple(map(int, [x3, y3]))
            p4 = tuple(map(int, [x4, y4]))

            points = np.array([p1, p2, p3, p4], dtype=np.int32)

            (x, y, w, h) = cv2.boundingRect(points)
            cutout = make_cutout(img, [x, y, x+w, y+h])
            cv2.imwrite(os.path.join(EXPORT_PATH, fil_name), img)

        except Exception as e:
            print(e)
            continue


