from argparse import ArgumentParser
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, BackboneFinetuning

from vox2vec.default_params import *
from vox2vec.eval.btcv import BTCV
from vox2vec.nn import FPN3d, FPNLinearHead, FPNNonLinearHead
from vox2vec.eval.end_to_end import EndToEnd
from vox2vec.eval.probing import Probing
from vox2vec.eval.svd_probing import SVD_Probing

from vox2vec.utils.misc import save_json
import numpy as np

def parse_args():
    parser = ArgumentParser()
    
    parser.add_argument('--dataset', default='btcv')
    parser.add_argument('--btcv_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/btcv', required=False)
    parser.add_argument('--cache_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/cache', required=False)
    parser.add_argument('--ckpt', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/pretrained_models/vox2vec.pt', required=False)
    # parser.add_argument('--setup', default='from_scratch', required=False)
    parser.add_argument('--setup', default='backbone_only', required=False)
    parser.add_argument('--log_dir', default='/media/jamshid/b0ad3209-9fa7-42e8-a070-b02947a78943/home/jamshid/git_clones/voxsep/vox2vec/logs', required=False)

    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--spacing', nargs='+', type=float, default=SPACING)
    parser.add_argument('--patch_size', nargs='+', type=int, default=PATCH_SIZE)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_batches_per_epoch', type=int, default=300)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=50)  # used only in finetuning setup

    parser.add_argument('--base_channels', type=int, default=BASE_CHANNELS)
    parser.add_argument('--num_scales', type=int, default=NUM_SCALES)

    return parser.parse_args()


def main(args):
    if args.dataset == 'btcv':
        datamodule = BTCV(
            root=args.btcv_dir,
            cache_dir=args.cache_dir,
            spacing=tuple(args.spacing),
            window_hu=WINDOW_HU,
            patch_size=tuple(args.patch_size),
            batch_size=args.batch_size,
            num_batches_per_epoch=args.num_batches_per_epoch,
            num_workers=args.num_workers,
            prefetch_factor=1,
            split=args.split,
        )
        num_classes = BTCV.num_classes
    else:
        raise NotImplementedError(f'Dataset {args.dataset} is not supported.')

    in_channels = 1
    backbone = FPN3d(in_channels, args.base_channels, args.num_scales)
    sd = torch.load(args.ckpt)
    backbone.load_state_dict(sd)
    if args.ckpt is not None:
        backbone.load_state_dict(torch.load(args.ckpt))
        backbone = backbone.cuda()

    if args.setup == 'probing': 
        heads = [
            FPNLinearHead(args.base_channels, args.num_scales, num_classes),
            FPNNonLinearHead(args.base_channels, args.num_scales, num_classes)
    ]
        head_linear = heads[0].cuda()

        i=0
        latents = []
        for batch in datamodule.train_dataloader():
            while i<2:    
                images = batch[0]
                latent = backbone(images.cuda(non_blocking=True))
                x_linear = head_linear(latent)
                print(i)
                latents.append(x_linear)
                i+=1
        latents = torch.cat(latents, dim=0)


    # model = Probing(backbone, *heads, patch_size=tuple(args.patch_size))
    # callbacks = [
    #     ModelCheckpoint(save_top_k=1, monitor='val/head_1_avg_dice_score', filename='best', mode='max'),
    # ]

    # logger = TensorBoardLogger(
    #     save_dir=args.log_dir,
    #     name=f'eval/{args.dataset}/{args.setup}/split_{args.split}'
    # )


    # i=0
    # latents = []
    # for batch in datamodule.train_dataloader():
    #     with torch.no_grad():
    #         while i<2:
                
    #             images = batch[0]
    #             output = model(images.cuda(non_blocking=True))
    #             print(i)
    #             latents.append(output[0])
    #             i+=1
    #     latents = torch.cat(latents, dim=0)


    if args.setup == 'backbone_only': 
        backbone = backbone.cuda()
        i=0
        latents = []
        for batch in datamodule.train_dataloader():
            while i<2:
                images = batch[0]
                output = backbone(images.cuda(non_blocking=True))
                
                print(i)
                latents.append(output[5].view(output[5].size(0), -1))
                i+=1
        latents = torch.cat(latents, dim=0)
        z = torch.nn.functional.normalize(latents, dim=1)
        z = z.cpu().detach().numpy()
        print('z.shape', z.shape)
        z = np.transpose(z)
        c = np.cov(z)
        print('c.shape', c.shape)
        # rank = np.linalg.matrix_rank(c, 1e-8)
        rank = np.linalg.matrix_rank(c)
        print('convariance matrix rank is', rank )
        _, d, _ = np.linalg.svd(c)
        print('d', d)

        
if __name__ == '__main__':
    main(parse_args())
