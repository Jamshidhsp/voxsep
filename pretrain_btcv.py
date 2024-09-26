from argparse import ArgumentParser

import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from vox2vec.default_params import *
from vox2vec.pretrain.data import PretrainDataset
from vox2vec.utils.data import ResizeByRandomSampling
from vox2vec.eval.btcv import BTCV
from vox2vec.nn import FPN3d, FPNLinearHead, FPNNonLinearHead, Reconstruction
from vox2vec.pretrain.model import Vox2Vec
# from vox2vec.pretrain.model_modified import Vox2Vec
# from vox2vec.eval.online_probing import OnlineProbing
import shutil
import os
import time


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--probing_dataset', default='btcv')
    parser.add_argument('--cache_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/cache', required=False)
    # parser.add_argument('--cache_dir', default='/home/jamshid/Datasets/cache/', required=False)
    
    parser.add_argument('--log_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/logs_btcv_to_btcv_single_volume', required=False)
    '''
    # parser.add_argument('--amos_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/datasets/BTCV_niigz/amos_template/', required=False)
    '''
    # parser.add_argument('--amos_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/server_datasets/amos', required=False)
    # parser.add_argument('--amos_dir', default='/home/jamshid/Datasets/', required=False)

    parser.add_argument('--amos_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/datasets/BTCV_niigz/amos_template_single_volume/', required=False)
    

    parser.add_argument('--flare_dir')
    parser.add_argument('--nlst_dir')
    parser.add_argument('--midrc_dir', default="/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/Datasets/midrc/manifest-1608266677008/", required=False)
    parser.add_argument('--nsclc_dir')
    # parser.add_argument('--btcv_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/btcv', required=False)
    '''
    parser.add_argument('--btcv_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/btcv_new', required=False)
    '''
    # parser.add_argument('--btcv_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/server_datasets/btcv', required=False)
    parser.add_argument('--btcv_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/btcv_single_volume', required=False)

    parser.add_argument('--spacing', nargs='+', type=float, default=SPACING)
    parser.add_argument('--patch_size', nargs='+', type=int, default=PATCH_SIZE)
    parser.add_argument('--pretrain_batch_size', type=int, default=2)
    parser.add_argument('--pretrain_num_workers', type=int, default=4)
    parser.add_argument('--probing_batch_size', type=int, default=2)
    parser.add_argument('--probing_num_workers', type=int, default=2)
    parser.add_argument('--num_batches_per_epoch', type=int, default=100)
    parser.add_argument('--val_every_n_epoch', type=int, default=2000)                                 

    parser.add_argument('--base_channels', type=int, default=BASE_CHANNELS)
    parser.add_argument('--num_scales', type=int, default=NUM_SCALES)

    return parser.parse_args()


def main(args):
    copy_time = time.time()

    # source = "/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/vox2vec/pretrain"
    # desination = "/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/runs_shutil/"+str(time.time())
    # if not os.path.exists(desination):
    #     os.mkdir(desination)
    # shutil.copytree(source, desination)
    spacing = tuple(args.spacing)
    patch_size = tuple(args.patch_size)
    pretrain_dataset = PretrainDataset(
        cache_dir=args.cache_dir,
        spacing=spacing,
        patch_size=patch_size,
        window_hu=WINDOW_HU,
        min_window_hu=MIN_WINDOW_HU,
        max_window_hu=MAX_WINDOW_HU,
        max_num_voxels_per_patch=MAX_NUM_VOXELS_PER_PATCH,
        batch_size=args.pretrain_batch_size,
        amos_dir=args.amos_dir,
        # flare_dir=args.flare_dir,
        # nlst_dir=args.nlst_dir,
        midrc_dir=args.midrc_dir,
        # nsclc_dir=args.nsclc_dir,
    )
    pretrain_dataset = ResizeByRandomSampling(pretrain_dataset, size=args.num_batches_per_epoch)
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=None,
        num_workers=args.pretrain_num_workers,
        prefetch_factor=16
    )

    in_channels = 1
    backbone = FPN3d(in_channels, args.base_channels, args.num_scales)
    # backbone = FPN3dUniform(in_channels, args.base_channels, args.num_scales)
    # backbone = Reconstruction(args.base_channels, args.num_scales)
    # model = Vox2Vec(
    #     backbone=backbone,
    #     base_channels=args.base_channels,
    #     num_scales=args.num_scales,
    # )
    model = Vox2Vec(
        # backbone=backbone,
        # base_channels=args.base_channels,
        # num_scales=args.num_scales,
    )

    # online probing
    if args.probing_dataset == 'btcv':
        probing_datamodule = BTCV(
            root=args.btcv_dir,
            cache_dir=args.cache_dir,
            spacing=spacing,
            window_hu=WINDOW_HU,
            patch_size=patch_size,
            batch_size=args.probing_batch_size,
            num_batches_per_epoch=args.num_batches_per_epoch,
            num_workers=args.probing_num_workers,
            prefetch_factor=16,
            split=0
        )
        num_classes = BTCV.num_classes
    else:
        raise NotImplementedError(f'Dataset {args.dataset} is not supported yet.')

    heads = [
        # FPNLinearHead(args.base_channels, args.num_scales, num_classes),
        FPNNonLinearHead(args.base_channels, args.num_scales, num_classes)
    ]
    # probing_callback = OnlineProbing(*heads, patch_size=patch_size)

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=args.log_dir, name='pretrain/'),
        # callbacks=[probing_callback],
        accelerator='gpu',
        max_epochs=-1,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.val_every_n_epoch,
        # accumulate_grad_batches=1,
        track_grad_norm=2
    )
    trainer.fit(
        model=model,
        train_dataloaders={
            'pretrain': pretrain_dataloader,
            # 'online_probing': probing_datamodule.train_dataloader()
        },
        val_dataloaders=probing_datamodule.val_dataloader(),
    )


if __name__ == '__main__':
    main(parse_args())
