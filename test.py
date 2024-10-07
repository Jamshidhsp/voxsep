# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from time import time
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
# from models.voco_head import VoCoHead
# from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
# from utils.ops import *
# from utils.utils import AverageMeter, distributed_all_gather
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))


# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Callable, Sequence
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import collections
import numpy as np
from monai.data import *
import pickle
from monai.transforms import *
from math import *


def get_loader(args):
    splits1 = "/btcv.json"
    splits2 = "/dataset_TCIAcovid19_0.json"
    splits3 = "/dataset_LUNA16_0.json"
    splits4 = "/stoic21.json"
    splits5 = "/Totalsegmentator_dataset.json"
    splits6 = "/flare23.json"
    splits7 = "/HNSCC.json"

    list_dir = "./jsons/"
    list_dir="/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voco/VoCo/jsons"
    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    jsonlist4 = list_dir + splits4
    jsonlist5 = list_dir + splits5
    jsonlist6 = list_dir + splits6
    jsonlist7 = list_dir + splits7

    datadir1 = "/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voco/VoCo/data/BTCV"


    num_workers = 4
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)


    vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
    # vallist2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
    # vallist3 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)

    datalist = new_datalist1 #+ datalist2 + new_datalist3 + datalist4 + datalist5 + datalist6 + datalist7
    val_files = vallist1 #+ vallist2 + vallist3  # + vallist4 + vallist5 + vallist6 + vallist7
    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))

    train_transforms = Compose([LoadImaged(keys=["image"], image_only=True, dtype=np.int16),
                                EnsureChannelFirstd(keys=["image"]),
                                Orientationd(keys=["image"], axcodes="RAS"),
                                ScaleIntensityRanged(
                                    keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                    b_min=args.b_min, b_max=args.b_max, clip=True),
                                CropForegroundd(keys="image", source_key="image", select_fn=threshold),
                                Resized(keys="image", mode="bilinear", align_corners=True,
                                        spatial_size=(384, 384, 96)),
                                VoCoAugmentation(args, aug=False)
                                ])

    # if args.cache_dataset:
    #     print("Using MONAI Cache Dataset")
    #     train_ds = CacheDataset(data=datalist, transform=train_transforms,
    #                             cache_rate=0.5, num_workers=num_workers)
    # elif args.smartcache_dataset:
    #     print("Using MONAI SmartCache Dataset")
    #     train_ds = SmartCacheDataset(
    #         data=datalist,
    #         transform=train_transforms,
    #         replace_rate=1.0,
    #         cache_num=2 * args.batch_size * args.sw_batch_size,
    #     )
    # else:
    #     print("Using Persistent dataset")
    #     train_ds = Dataset(data=datalist, transform=train_transforms)
        # train_ds = PersistentDataset(data=datalist,
        #                              transform=train_transforms,
        #                              pickle_protocol=pickle.HIGHEST_PROTOCOL,
        #                              cache_dir='data/linshan/cache/10k')

    # if args.distributed:
    #     train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    # else:
    train_ds = Dataset(data=datalist, transform=train_transforms)
    train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, shuffle=True,
        drop_last=True, pin_memory=True
    )

    return train_loader


def threshold(x):
    # threshold at 0
    return x > 0.3


class VoCoAugmentation():
    def __init__(self, args, aug):
        self.args = args
        self.aug = aug

    def __call__(self, x_in):
        crops_trans = get_crop_transform(roi_small=self.args.roi_x, aug=self.aug)

        vanilla_trans, labels = get_vanilla_transform(num=self.args.sw_batch_size,
                                                      roi_small=self.args.roi_x, aug=self.aug)

        imgs = []
        for trans in vanilla_trans:
            img = trans(x_in)
            imgs.append(img)

        crops = []
        for trans in crops_trans:
            crop = trans(x_in)
            crops.append(crop)

        return imgs, labels, crops


def get_vanilla_transform(num=2, num_crops=4, roi_small=64, roi=96, max_roi=384, aug=False):
    vanilla_trans = []
    labels = []
    for i in range(num):
        center_x, center_y, label = get_position_label(roi=roi,
                                                       max_roi=max_roi,
                                                       num_crops=num_crops)
        if aug:
            trans = Compose([
                SpatialCropd(keys=['image'],
                             roi_center=[center_x, center_y, roi // 2],
                             roi_size=[roi, roi, roi]),
                Resized(keys=["image"], mode="bilinear", align_corners=True,
                        spatial_size=(roi_small, roi_small, roi_small)),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                ToTensord(keys=["image"])])
        else:
            trans = Compose([
                SpatialCropd(keys=['image'],
                             roi_center=[center_x, center_y, roi // 2],
                             roi_size=[roi, roi, roi]),
                Resized(keys=["image"], mode="bilinear", align_corners=True,
                        spatial_size=(roi_small, roi_small, roi_small)),
                ToTensord(keys=["image"])])

        vanilla_trans.append(trans)
        labels.append(label)

    labels = np.concatenate(labels, 0).reshape(num, num_crops * num_crops)

    return vanilla_trans, labels


def get_crop_transform(num=4, roi_small=64, roi=96, aug=False):
    voco_trans = []
    # not symmetric at axis x !!!
    for i in range(num):
        for j in range(num):
            center_x = (i + 1 / 2) * roi
            center_y = (j + 1 / 2) * roi
            center_z = roi // 2

            if aug:
                trans = Compose([
                    SpatialCropd(keys=['image'],
                                 roi_center=[center_x, center_y, center_z],
                                 roi_size=[roi, roi, roi]),
                    Resized(keys=["image"],
                            mode="bilinear",
                            align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small)
                            ),
                    Resized(keys=["image"], mode="bilinear", align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small)),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                    RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                    ToTensord(keys=["image"])],
                )
            else:
                trans = Compose([
                    SpatialCropd(keys=['image'],
                                 roi_center=[center_x, center_y, center_z],
                                 roi_size=[roi, roi, roi]),
                    Resized(keys=["image"],
                            mode="bilinear",
                            align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small)
                            ),
                    ToTensord(keys=["image"])],
                )

            voco_trans.append(trans)

    return voco_trans


def get_position_label(roi=96, base_roi=96, max_roi=384, num_crops=4):
    half = roi // 2
    center_x, center_y = np.random.randint(low=half, high=max_roi - half), \
        np.random.randint(low=half, high=max_roi - half)

    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half

    total_area = roi * roi
    labels = []
    for i in range(num_crops):
        for j in range(num_crops):
            crop_x_min, crop_x_max = i * base_roi, (i + 1) * base_roi
            crop_y_min, crop_y_max = j * base_roi, (j + 1) * base_roi

            dx = min(crop_x_max, x_max) - max(crop_x_min, x_min)
            dy = min(crop_y_max, y_max) - max(crop_y_min, y_min)
            if dx <= 0 or dy <= 0:
                area = 0
            else:
                area = (dx * dy) / total_area
            labels.append(area)

    labels = np.asarray(labels).reshape(1, num_crops * num_crops)

    return center_x, center_y, labels


if __name__ == '__main__':
    center_x, center_y, labels = get_position_label()
    print(center_x, center_y, labels)













def main():

    roi = 64
    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="logs", type=str, help="directory to save logs")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=250000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=1, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=5000, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    # './runs/logs_10k/model_current_epoch.pt'
    parser.add_argument("--resume", default=None, type=str,
                        help="resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", default=True, help="do NOT use amp for training")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--smartcache_dataset", default=False, help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")

    args = parser.parse_args()
    # logdir = args.logdir



    train_loader = get_loader(args)

    best_val = 1e8

    scaler = None


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
    main()
