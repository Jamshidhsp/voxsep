from argparse import ArgumentParser

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from vox2vec.default_params import *
from vox2vec.pretrain.data import PretrainDataset
from vox2vec.utils.data import VanillaDataset, ResizeByRandomSampling
from vox2vec.eval.btcv import BTCV
from vox2vec.nn import FPN3d, FPNLinearHead, FPNNonLinearHead
from vox2vec.pretrain.model import Vox2Vec
from vox2vec.eval.online_probing import OnlineProbing



def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--probing_dataset', default='btcv')
    parser.add_argument('--cache_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/cache', required=False)
    parser.add_argument('--log_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/logs', required=False)

    parser.add_argument('--amos_dir')
    parser.add_argument('--flare_dir')
    parser.add_argument('--nlst_dir')
    parser.add_argument('--midrc_dir')
    parser.add_argument('--nsclc_dir')
    parser.add_argument('--btcv_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/btcv', required=False)

    parser.add_argument('--spacing', nargs='+', type=float, default=SPACING)
    parser.add_argument('--patch_size', nargs='+', type=int, default=PATCH_SIZE)
    parser.add_argument('--pretrain_batch_size', type=int, default=10)
    parser.add_argument('--pretrain_num_workers', type=int, default=10)
    parser.add_argument('--probing_batch_size', type=int, default=5)
    parser.add_argument('--probing_num_workers', type=int, default=1)
    parser.add_argument('--num_batches_per_epoch', type=int, default=100)
    parser.add_argument('--val_every_n_epoch', type=int, default=10)

    parser.add_argument('--base_channels', type=int, default=BASE_CHANNELS)
    parser.add_argument('--num_scales', type=int, default=NUM_SCALES)

    return parser.parse_args()


def main(args):
    spacing = tuple(args.spacing)
    patch_size = tuple(args.patch_size)

    pretrain_dataset = BTCV(
            root=args.btcv_dir,
            cache_dir=args.cache_dir,
            spacing=spacing,
            window_hu=WINDOW_HU,
            patch_size=patch_size,
            batch_size=args.pretrain_batch_size,
            num_batches_per_epoch=1,
            num_workers=2,
            prefetch_factor=16,
            split=0,
        )

    # pretrain_dataset = ResizeByRandomSampling(pretrain_dataset, size=args.num_batches_per_epoch)
    # pretrain_dataloader = DataLoader(
    #     pretrain_dataset,
    #     batch_size=None,
    #     num_workers=args.pretrain_num_workers,
    #     prefetch_factor=16
    # )
    pretrain_dataloader = pretrain_dataset.train_dataloader()
    in_channels = 1
    backbone = FPN3d(in_channels, args.base_channels, args.num_scales)
    model = Vox2Vec(
        backbone=backbone,
        base_channels=args.base_channels,
        num_scales=args.num_scales,
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
        FPNLinearHead(args.base_channels, args.num_scales, num_classes),
        FPNNonLinearHead(args.base_channels, args.num_scales, num_classes)
    ]
    probing_callback = OnlineProbing(*heads, patch_size=patch_size)

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=args.log_dir, name='pretrain/'),
        callbacks=[probing_callback],
        accelerator='gpu',
        max_epochs=-1,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.val_every_n_epoch
    )
    trainer.fit(
        model=model,
        train_dataloaders={
            'pretrain': pretrain_dataloader,
            'online_probing': probing_datamodule.train_dataloader()
        },
        val_dataloaders=probing_datamodule.val_dataloader(),
    )


if __name__ == '__main__':
    main(parse_args())
