"""
Borrowed from https://github.com/hysts/pytorch_image_classification/tree/9ff4248905850c68aa9c09c17914307eb81769e7/pytorch_image_classification/transforms
"""
import torch
import numpy as np
import PIL
import PIL.Image
from PIL.Image import Image


class NpNormalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image: PIL.Image.Image) -> np.ndarray:
        image = np.asarray(image).astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


#
# class Cutout:
#     def __init__(self, p=1.0, mask_size=16, cutout_inside=False, mask_color=0):
#         # https://github.com/hysts/pytorch_image_classification/blob/9ff4248905850c68aa9c09c17914307eb81769e7/configs/augmentations/cifar/cutout.yaml
#         self.p = p
#         self.mask_size = mask_size
#         self.cutout_inside = cutout_inside
#         self.mask_color = mask_color
#
#         self.mask_size_half = self.mask_size // 2
#         self.offset = 1 if self.mask_size % 2 == 0 else 0
#
#     def __call__(self, image: np.ndarray) -> np.ndarray:
#         image = np.asarray(image).copy()
#
#         if np.random.random() > self.p:
#             return image
#
#         h, w = image.shape[:2]
#
#         if self.cutout_inside:
#             cxmin = self.mask_size_half
#             cxmax = w + self.offset - self.mask_size_half
#             cymin = self.mask_size_half
#             cymax = h + self.offset - self.mask_size_half
#         else:
#             cxmin, cxmax = 0, w + self.offset
#             cymin, cymax = 0, h + self.offset
#
#         cx = np.random.randint(cxmin, cxmax)
#         cy = np.random.randint(cymin, cymax)
#         xmin = cx - self.mask_size_half
#         ymin = cy - self.mask_size_half
#         xmax = xmin + self.mask_size
#         ymax = ymin + self.mask_size
#         xmin = max(0, xmin)
#         ymin = max(0, ymin)
#         xmax = min(w, xmax)
#         ymax = min(h, ymax)
#         image[ymin:ymax, xmin:xmax] = self.mask_color
#         return image


class RandomErasing:
    def __init__(self, p=0.5, max_attempt=20, sl=0.02, sh=0.4, rl=0.3, rh=1. / 0.3):
        # https://github.com/hysts/pytorch_image_classification/blob/9ff4248905850c68aa9c09c17914307eb81769e7/configs/augmentations/cifar/random_erasing.yaml
        self.p = 0.5
        self.max_attempt = 20
        self.sl, self.sh = 0.02, 0.4
        self.rl = 0.3
        self.rh = 1. / 0.3

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image).copy()

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(self.max_attempt):
            mask_area = np.random.uniform(self.sl, self.sh) * image_area
            aspect_ratio = np.random.uniform(self.rl, self.rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)
                break

        return image
