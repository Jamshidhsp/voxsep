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
# jamshid was here
import math
import torch.nn.functional as F


def prepare_nlst_ids(nlst_dir, patch_size):
    nlst = NLST(root=nlst_dir)
    for id_ in tqdm(nlst.ids, desc='Warming up NLST().patient_id method'):
        nlst.patient_id(id_)

    nlst_patients = nlst >> GroupBy('patient_id')
    ids = []
    for patient_id in tqdm(nlst_patients.ids, desc='Preparing NLST ids'):
        id, slice_locations = max(nlst_patients.slice_locations(patient_id).items(), key=lambda i: len(i[1]))
        if len(slice_locations) >= patch_size[2]:
            ids.append(id)

    return ids


class PretrainDataset(Dataset):
    def __init__(
            self,
            cache_dir: str,
            spacing: Tuple[float, float, float],
            patch_size: Tuple[int, int, int],
            window_hu: Tuple[float, float],
            min_window_hu: Tuple[float, float],
            max_window_hu: Tuple[float, float],
            max_num_voxels_per_patch: int,
            batch_size: int,
            amos_dir: Optional[str] = None,
            flare_dir: Optional[str] = None,
            nlst_dir: Optional[str] = None,
            midrc_dir: Optional[str] = None,
            nsclc_dir: Optional[str] = None,
    ) -> None:
        parse_affine = Transform(
            __inherit__=True,
            flipped_axes=lambda affine: tuple(np.where(np.diag(affine[:3, :3]) < 0)[0] - 3),  # enumerate from the end
            spacing=lambda affine: tuple(np.abs(np.diag(affine[:3, :3]))),
        )

        # amos_ct_ids = AMOS(root=amos_dir).ids[:500]
        amos_ct_ids = AMOS(root=amos_dir).ids[:]
        amos = Chain(
            AMOS(root=amos_dir),
            Filter.keep(amos_ct_ids),
            parse_affine,
            FlipAxesToCanonical(),
        )

        # flare = Chain(
        #     FLARE2022(root=flare_dir),
        #     Filter(lambda id: id.startswith('TU'), verbose=True),
        #     Filter(lambda affine: is_diagonal(affine[:3, :3]), verbose=True),
        #     CacheToDisk.simple('ids', root=cache_dir),
        #     parse_affine,
        #     FlipAxesToCanonical(),
        # )

        # nlst = Chain(
        #     NLST(root=nlst_dir),
        #     Transform(__inherit__=True, ids=lambda: prepare_nlst_ids(nlst_dir, patch_size)),
        #     CacheToDisk.simple('ids', root=cache_dir),
        #     LocationsToSpacing(),
        #     Apply(image=lambda x: np.flip(x, axis=(0, 1)).copy())
        # )

        # midrc = Chain(
        #     MIDRC(root=midrc_dir),
        #     Apply(image=lambda x: np.flip(x, axis=(0, 1)).copy())
        # )

        # nsclc = Chain(
        #     NSCLC(root=nsclc_dir),
        #     Apply(image=lambda x: np.flip(x, axis=(0, 1)).copy())
        # )

        # lidc = Chain(
        #     LIDC(),  # see amid docs
        #     Apply(image=lambda x: np.flip(np.swapaxes(x, 0, 1), axis=(0, 1)).copy())
        # )

        # use connectome for smart cashing (with automatic invalidation)
        pipeline = Chain(
            Merge(
                amos,  # 500 abdominal CTs
                # flare,  # 2000 abdominal CTs
                # nlst,  # ~2500 thoracic CTs
                # midrc,  # ~150 thoracic CTs (most patients with COVID-19)
                # nsclc,  # ~400 thoracic CTs (most patients with severe non-small cell lung cancer)
                # lidc  # ~1000 thoracic CTs (most patients with lung nodules)
            ),  # ~6550 openly available CTs in total, covering abdomen and thorax domains
            # cache spacing
            CacheToDisk.simple('spacing', root=cache_dir),
            Filter(lambda spacing: spacing[-1] is not None, verbose=True),
            CacheToDisk.simple('ids', root=cache_dir),
            # cropping, rescaling
            Transform(__inherit__=True, cropping_box=lambda image: mask_to_bbox(image >= BODY_THRESHOLD_HU)),
            CropToBox(axis=(-3, -2, -1)),
            RescaleToSpacing(to_spacing=spacing, axis=(-3, -2, -1), image_fill_value=lambda x: np.min(x)),
            Apply(image=lambda x: np.int16(x)),
            CacheToDisk.simple('image', root=cache_dir),
            Apply(image=lambda x: np.float32(x)),
            # filtering by shape
            Filter(lambda image: np.all(np.array(image.shape) >= patch_size), verbose=True),
            CacheToDisk.simple('ids', root=cache_dir),
            # adding body_voxels
            Transform(__inherit__=True, body_voxels=lambda image: np.argwhere(get_body_mask(image))),
            CacheToDisk.simple('body_voxels', root=cache_dir),
            Filter(lambda body_voxels: len(body_voxels) > 0, verbose=True),
            CacheToDisk.simple('ids', root=cache_dir),
        )

        self.pipeline = pipeline
        self.ids = pipeline.ids
        self.load_example = pipeline._compile(['image', 'body_voxels'])
        self.patch_size = patch_size
        self.window_hu = window_hu
        self.min_window_hu = min_window_hu
        self.max_window_hu = max_window_hu
        self.max_num_voxels_per_patch = max_num_voxels_per_patch
        self.batch_size = batch_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        args = [*self.load_example(self.ids[i]), self.patch_size,
                self.window_hu, self.min_window_hu, self.max_window_hu,
                self.max_num_voxels_per_patch]
        views = [sample_views(*args) for _ in range(self.batch_size)]
        patches_1, patches_2, voxels_1, voxels_2 = zip(*views)
        patches_1 = torch.tensor(np.stack([p[None] for p in patches_1]))
        patches_2 = torch.tensor(np.stack([p[None] for p in patches_2]))
        voxels_1 = [torch.tensor(voxels) for voxels in voxels_1]
        voxels_2 = [torch.tensor(voxels) for voxels in voxels_2]
        return patches_1, patches_2, voxels_1, voxels_2


def sample_views(
        image: np.ndarray,
        roi_voxels: np.ndarray,
        patch_size: Tuple[int, int, int],
        window_hu: Tuple[float, float],
        min_window_hu: Tuple[float, float],
        max_window_hu: Tuple[float, float],
        max_num_voxels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_voxel = random.choice(roi_voxels)  # (3,)

    # rotate_angle = random.choice(0, 30)

    patch_1, roi_voxels_1 = sample_view(image, roi_voxels, anchor_voxel, patch_size,
                                         window_hu, min_window_hu, max_window_hu)
    patch_2, roi_voxels_2 = sample_view(image, roi_voxels, anchor_voxel, patch_size,
                                         window_hu, min_window_hu, max_window_hu)
    
    # patch_1, roi_voxels_1 = sample_view(image, roi_voxels, anchor_voxel, patch_size,
    #                                      window_hu, min_window_hu, max_window_hu, rotate_angle)
    # patch_2, roi_voxels_2 = sample_view(image, roi_voxels, anchor_voxel, patch_size,
    #                                      window_hu, min_window_hu, max_window_hu, rotate_angle)

    valid_1 = np.all((roi_voxels_1 >= 0) & (roi_voxels_1 < patch_size), axis=1)
    valid_2 = np.all((roi_voxels_2 >= 0) & (roi_voxels_2 < patch_size), axis=1)
    valid = valid_1 & valid_2
    assert valid.any()
    indices = np.where(valid)[0]

    if len(indices) > max_num_voxels:
        indices = np.random.choice(indices, max_num_voxels, replace=False)

    return patch_1, patch_2, roi_voxels_1[indices], roi_voxels_2[indices]


def sample_view(image, voxels, anchor_voxel, patch_size, window_hu, min_window_hu, max_window_hu):
# def sample_view(image, voxels, anchor_voxel, patch_size, window_hu, min_window_hu, max_window_hu, rotate_angle):
    assert image.ndim == 3

    # spatial augmentations: random rescale, rotation and crop
    box = sample_box(image.shape, patch_size, anchor_voxel)
    image = crop_to_box(image, box, axis=(-3, -2, -1))
    shift = box[0]
    voxels = voxels - shift
    anchor_voxel = anchor_voxel - shift
    # return image, voxels

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
    if random.uniform(0, 1) < -10.5:
        angle = 5
        image = rotate_volume(image, angle, axis='z')
        voxels = rotate_voxels(voxels, angle, axis='z')
    
    
    return image, voxels




import numpy as np
import math
from scipy.ndimage import affine_transform

def rotate_volume(image, angle, axis='z'):
    angle_rad = math.radians(angle)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    else:  # axis == 'z'
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    rotated_image = affine_transform(image, rotation_matrix)

    return rotated_image



import numpy as np
import math

def rotate_voxels(point, angle, axis='z'):

    angle_rad = math.radians(angle)

    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    else:  # axis == 'z'
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    rotated_point = np.dot(rotation_matrix, point.T).T

    return rotated_point






# def rotate_volume(image, angle, axis='z'):

#     # Convert angle to radians
#     angle_rad = math.radians(angle)

#     # Create affine transformation matrix for 3D rotation
#     # Adjusted to a 3x4 matrix format
#     if axis == 'x':
#         affine_matrix = torch.tensor([
#             [1, 0, 0, 0],
#             [0, math.cos(angle_rad), -math.sin(angle_rad), 0],
#             [0, math.sin(angle_rad), math.cos(angle_rad), 0]
#         ], dtype=torch.float32)

#     elif axis == 'y':
#         affine_matrix = torch.tensor([
#             [math.cos(angle_rad), 0, math.sin(angle_rad), 0],
#             [0, 1, 0, 0],
#             [-math.sin(angle_rad), 0, math.cos(angle_rad), 0]
#         ], dtype=torch.float32)

#     else:  # 'z'
#         affine_matrix = torch.tensor([
#             [math.cos(angle_rad), -math.sin(angle_rad), 0, 0],
#             [math.sin(angle_rad), math.cos(angle_rad), 0, 0],
#             [0, 0, 1, 0]
#         ], dtype=torch.float32)

#     image = image.unsqueeze(0).unsqueeze(0)

#     # Create grid for affine transformation
#     D, H, W = image.shape[2], image.shape[3], image.shape[4]
#     grid_size = torch.Size([1, 1, D, H, W])
#     grid = F.affine_grid(affine_matrix.unsqueeze(0), grid_size, align_corners=True)

#     # Apply the transformation
#     rotated_image = F.grid_sample(image, grid, align_corners=True)

#     # Remove added batch and channel dimensions
#     rotated_image = rotated_image.squeeze(0).squeeze(0)

#     return rotated_image




# def rotate_voxels(voxels, angle, axis='z'):
#     angle_rad = math.radians(angle)
#     rot_x = torch.tensor([
#         [1, 0, 0],
#         [0, math.cos(angle_rad), -math.sin(angle_rad)],
#         [0, math.sin(angle_rad), math.cos(angle_rad)]
#     ])
#     rot_y = torch.tensor([
#         [math.cos(angle_rad), 0, math.sin(angle_rad)],
#         [0, 1, 0],
#         [-math.sin(angle_rad), 0, math.cos(angle_rad)]
#     ])
#     rot_z = torch.tensor([
#         [math.cos(angle_rad), -math.sin(angle_rad), 0],
#         [math.sin(angle_rad), math.cos(angle_rad), 0],
#         [0, 0, 1]
#     ])
#     if axis == 'x':
#         rotation_matrix = rot_x
#     elif axis == 'y':
#         rotation_matrix = rot_y
#     else:
#         rotation_matrix = rot_z    
#     rotated_voxels = torch.matmul(rotation_matrix, voxels)

#     return rotated_voxels

