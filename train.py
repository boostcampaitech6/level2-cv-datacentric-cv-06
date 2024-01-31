import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

import random
import numpy as np

import wandb
from datetime import datetime

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../data/medical'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--ignore_tags', type=list, default=['masked', 'excluded-region', 'maintable', 'stamp'])
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def denormalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    
    # Convert mean and std to tensors
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(image.device)

    denormalized_image = (image * std) + mean

    # Remove batch dimension if present
    denormalized_image = denormalized_image.squeeze(0)

    return denormalized_image

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, ignore_tags, seed):
    dataset = SceneTextDataset(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / batch_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 140], gamma=0.1)
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.train()
    for epoch in range(max_epoch):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)
                
                # visualization
                vis_info = extra_info.copy()
                smap_to_log = [wandb.Image(smap.detach().cpu().numpy().transpose(1, 2, 0).squeeze()) for smap in vis_info['score_map']]
                geo_t_to_log = [wandb.Image(geo.detach().cpu().numpy().transpose(1, 2, 0)[:, :, 0]) for geo in vis_info['geo_map']]
                geo_b_to_log = [wandb.Image(geo.detach().cpu().numpy().transpose(1, 2, 0)[:, :, 1]) for geo in vis_info['geo_map']]
                geo_l_to_log = [wandb.Image(geo.detach().cpu().numpy().transpose(1, 2, 0)[:, :, 2]) for geo in vis_info['geo_map']]
                geo_r_to_log = [wandb.Image(geo.detach().cpu().numpy().transpose(1, 2, 0)[:, :, 3]) for geo in vis_info['geo_map']]
                denormalized_images = [denormalize(image) for image in img]
                images_to_log = [wandb.Image(image.permute(1, 2, 0).cpu().numpy()) for image in denormalized_images]
                
                table = wandb.Table(columns=["train_img", "score_map", "geo_top", "geo_bottom", "geo_left", "geo_right"],
                                    data=[[i,s,gt,gb,gl,gr] for i,s,gt,gb,gl,gr in zip(images_to_log, smap_to_log, geo_t_to_log, geo_b_to_log, geo_l_to_log, geo_r_to_log)])
                wandb.log({'Train Loss': loss_val, 'Cls Loss': extra_info['cls_loss'],
                    'Angle Loss': extra_info['angle_loss'], 'IoU Loss': extra_info['iou_loss'],
                    'Train Images': table})

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)
            
            os.makedirs(os.path.join(model_dir, train_serial), exist_ok=True)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
            }
            ckpt_fpath = osp.join(model_dir, train_serial, f'{epoch+1}.pth')
            torch.save(state, ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    wandb.init(project="CV06_Data_Centric",
               name="test",
               notes="",
               config={
                    "batch_size": args.batch_size,
                    "Learning_rate": args.learning_rate,
                    "Epochs": args.max_epoch,
                })
    wandb.run.save()
    
    main(args)
