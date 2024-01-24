from typing import *
from tqdm import tqdm
import numpy as np
import random
from imops import crop_to_box

import torch
from torch.utils.data import Dataset

from amid.amos import AMOS
from amid.flare2022 import FLARE2022
from amid.nlst import NLST
from amid.lidc import LIDC
from amid.nsclc import NSCLC
from amid.midrc import MIDRC

from connectome import Chain, Transform, Filter, Apply, GroupBy, Merge, CacheToDisk

from vox2vec.processing import (
    LocationsToSpacing, FlipAxesToCanonical, CropToBox, RescaleToSpacing,
    get_body_mask, BODY_THRESHOLD_HU, sample_box, gaussian_filter, gaussian_sharpen,
    scale_hu
)
from vox2vec.utils.box import mask_to_bbox
from vox2vec.utils.misc import is_diagonal


import numpy as np

WINDOW_HU = -1350, 1000

# pre-training
MIN_WINDOW_HU = -1000, 300
MAX_WINDOW_HU = -1350, 1000

def sample_view(image,window_hu = WINDOW_HU, min_window_hu=MIN_WINDOW_HU, max_window_hu=MAX_WINDOW_HU):
    # assert image.ndim == 3
    # intensity augmentations
    if random.uniform(0, 1) < 0.5:
        if random.uniform(0, 1) < 0.5:
            # random gaussian blur in axial plane
            sigma = random.uniform(0.25, 1.5)
            image = gaussian_filter(image, sigma, axis=(0, 1))
        else:
            # random gaussian sharpening in axial plane
            sigma_1 = random.uniform(0.5, 1.0)
            sigma_2 = 0.5
            alpha = random.uniform(10.0, 30.0)
            image = gaussian_sharpen(image, sigma_1, sigma_2, alpha, axis=(0, 1))

    if random.uniform(0, 1) < 0.5:
        sigma_hu = random.uniform(0, 30)
        image = image + np.random.normal(0, sigma_hu, size=image.shape).astype('float32')

    if random.uniform(0, 1) < 0.8:
        window_hu = (random.uniform(max_window_hu[0], min_window_hu[0]),
                     random.uniform(min_window_hu[1], max_window_hu[1]))
    image = scale_hu(image, window_hu)

# rotate
    if random.uniform(0, 1) > -10.5:
        angle = 5
        image = rot_rand(image)
    return image

import numpy
# there might be a need to batch-compatibility
def rot_rand(x):
    # x = x.detach().clone()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    orientation = np.random.randint(0, 4) *0
    if orientation == 0:
        pass
    elif orientation == 1:
        x = numpy.rot90(x, 1, (1, 2))
    elif orientation == 2:
        x = numpy.rot90(x, 2, (1, 2))
    elif orientation == 3:
        x = numpy.rot90(x, 3, (1, 2))
    return x



# def get_random_crops(image, crop_size, num_crops):
#     crops = []
    
#     for _ in range(num_crops):
#         start_x = np.random.randint(0, image.shape[0] - crop_size[0])
#         start_y = np.random.randint(0, image.shape[1] - crop_size[1])
#         start_z = np.random.randint(0, image.shape[2] - crop_size[2])
#         crop = image[start_x:start_x + crop_size[0], start_y:start_y + crop_size[1], start_z:start_z + crop_size[2]]
#         crops.append(crop)
#     return crops

# # Assuming `image1` and `image2` are your input patches with shape (128, 128, 32)
# image1 = np.random.rand(128, 128, 32)  # Replace with actual image data
# image2 = np.random.rand(128, 128, 32)  # Replace with actual image data

# crop_size = (32, 32, 8)  # Example crop size
# num_crops = 10  # Number of crops

# crops_image1 = get_random_crops(image1, crop_size, num_crops)
# crops_image2 = get_random_crops(image2, crop_size, num_crops)

# Crops from the same index are positive pairs, different indices are negative pairs


import torch

def get_non_overlapping_crops_and_pairs(image_tensor, crop_size, num_crops):
    """
    Get non-overlapping crops from the image tensor and create negative pairs.

    Parameters:
    image_tensor (torch.Tensor): The input image tensor.
    crop_size (tuple): The size of the crop (height, width, depth).
    num_crops (int): The number of crops to generate.

    Returns:
    List[torch.Tensor]: List of cropped images.
    List[tuple]: List of tuples representing negative pairs.
    """
    crops = []
    negative_pairs = []

    # Calculate the number of possible crops along each dimension
    num_possible_crops = (
        image_tensor.shape[0] // crop_size[0],
        image_tensor.shape[1] // crop_size[1],
        image_tensor.shape[2] // crop_size[2]
    )

    # Generate crops
    for i in range(num_crops):
        x = (i % num_possible_crops[0]) * crop_size[0]
        y = ((i // num_possible_crops[0]) % num_possible_crops[1]) * crop_size[1]
        z = (i // (num_possible_crops[0] * num_possible_crops[1])) * crop_size[2]
        crop = image_tensor[x:x + crop_size[0], y:y + crop_size[1], z:z + crop_size[2]]
        crops.append(crop)

    # Generate negative pairs (every crop with every other crop)
    for i in range(len(crops)):
        for j in range(len(crops)):
            if i != j:
                negative_pairs.append((crops[i], crops[j]))

    return crops, negative_pairs

# # Example usage
# # Assuming `image_tensor` is your input patch as a PyTorch tensor with shape (128, 128, 32)
# image_tensor = torch.rand(128, 128, 32)  # Replace with your actual image tensor

# crop_size = (32, 32, 8)  # Example crop size
# num_crops = 10  # Number of crops to generate (make sure it's feasible with your image and crop size)

# # Generate non-overlapping crops and negative pairs
# crops, negative_pairs = get_non_overlapping_crops_and_pairs(image_tensor, crop_size, num_crops)

# # Verify the number of crops and negative pairs
# len(crops), len(negative_pairs)


def get_non_overlapping_crops(image_tensor, crop_size):

    crops = image_tensor.unfold(0, crop_size[0], crop_size[0]) \
                        .unfold(1, crop_size[1], crop_size[1]) \
                        .unfold(2, crop_size[2], crop_size[2])
    return crops.reshape(-1, *crop_size)
