import numpy as np
from imgaug import augmenters as iaa


def build_pipeline():
    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.SomeOf((2, 4),
                       [
                           iaa.GaussianBlur((0.1, 2.0)),  # blur images with a sigma between 0 and 3.0
                           iaa.GammaContrast((0.5, 2.0)),
                           iaa.Add((-40, 40)),
                           iaa.SaltAndPepper(0.1),


                           # iaa.OneOf(
                           #  [
                           #     iaa.GaussianBlur((0.1, 2.0)),  # blur images with a sigma between 0 and 3.0
                           #     iaa.AverageBlur(k=(2, 5)),  # blur image using local means with kernel sizes between 2 and 7
                           #     iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
                           #  ]),

                           # iaa.OneOf(
                           #     [
                           #         iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
                           #         iaa.Add((-40, 40)),
                           #         # iaa.SaltAndPepper(0.1),
                           #         iaa.imgcorruptlike.MotionBlur(severity=2)
                           #         # iaa.imgcorruptlike.Fog(severity=2)
                           #     ]
                           # ),

                           # iaa.JpegCompression(compression=(0, 20))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )
    return seq


def augment_image(imgs, seq):
    aug_imgs = seq(images=imgs)
    # aug_imgs = imgs
    aug_imgs_norm = np.array((aug_imgs - np.min(aug_imgs)) / (np.max(aug_imgs) - np.min(aug_imgs)), dtype=np.float32)
    return aug_imgs_norm


def test_augment_image(imgs, seq):
    aug_imgs = seq(images=imgs)
    return aug_imgs

