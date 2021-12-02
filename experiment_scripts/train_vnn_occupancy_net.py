# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import summaries
import losses
import torch
import vnn_occupancy_network
import training
import dataio
import configargparse
from torch.utils.data import DataLoader
import numpy as np
import datetime

import torch.multiprocessing as mp
import torch.distributed as dist


def worker_init_fn(worker_id):
    #print(torch.utils.data.get_worker_info().seed)
    #print(torch.initial_seed())
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))

def sync_model(model):
    size = float(dist.get_world_size())

    for param in model.parameters():
        dist.broadcast(param.data, 0)

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='/home/jiahui/shape_occupancy/experiment_scripts/logs', help='root for logging')
p.add_argument('--dataset', type=str, default='joint',
               help='rack, bottle, mug, bowl, joint')
p.add_argument('--experiment_name',default=str(datetime.datetime.now()), type=str, required=True,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

p.add_argument('--sidelength', type=int, default=128)

# General training options
p.add_argument('--batch_size', type=int, default=64)
p.add_argument('--gpus', type=int, default=1, help='number of gpus')
p.add_argument('--lr', type=float, default=5e-4, help='learning rate. default=5e-5')
p.add_argument('--num_epochs', type=int, default=40001,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--iters_til_ckpt', type=int, default=10000,
               help='Training steps until save checkpoint')

p.add_argument('--depth_aug', action='store_true', help='depth_augmentation')
p.add_argument('--multiview_aug', action='store_false', help='multiview_augmentation')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
opt = p.parse_args()
# print(opt)

def main(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:6007', world_size=opt.gpus, rank=gpu)
    if opt.dataset == "bottle":
        train_dataset = dataio.BottleOccTrainDataset(128)
        val_dataset = dataio.BottleOccTrainDataset(128, phase='val')
    elif opt.dataset == "reprl":
        train_dataset = dataio.RepBCTrainDataset(128)
        val_dataset = dataio.RepBCTrainDataset(128, phase='val')
    elif opt.dataset == "bowl":
        train_dataset = dataio.BowlOccTrainDataset(128)
        val_dataset = dataio.BowlOccTrainDataset(128, phase='val')
    elif opt.dataset == "mug":
        train_dataset = dataio.DepthOccTrainDataset(128)
        val_dataset = dataio.DepthOccTrainDataset(128, phase='val')
    elif opt.dataset == "rack":
        train_dataset = dataio.RackOccTrainDataset(128)
        val_dataset = dataio.RackOccTrainDataset(128, phase='val')
    elif opt.dataset == "joint":
        train_dataset = dataio.JointOccTrainDataset(128, depth_aug=opt.depth_aug, multiview_aug=opt.multiview_aug)
        val_dataset = dataio.JointOccTrainDataset(128, phase='val')
    else:
        assert False

    torch.cuda.set_device(gpu)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  drop_last=True, num_workers=0, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True,
                                drop_last=True, num_workers=12, worker_init_fn=worker_init_fn)

    model = vnn_occupancy_network.VNNOccNet(latent_dim=128).cuda()

    if opt.checkpoint_path is not None:
        model.load_state_dict(torch.load(opt.checkpoint_path))

    # model_parallel = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model_parallel = model
    sync_model(model)

    # Define the loss
    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    # Define the loss
    summary_fn = summaries.occupancy_net
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    loss_fn = val_loss_fn = losses.occupancy_net

    # devices = torch.get_all_devices()
    count = torch.cuda.device_count()
    print("Devices",torch.cuda.is_available(),torch.cuda.device_count())
    training.train(model=model_parallel, train_dataloader=train_dataloader, val_dataloader=val_dataloader, epochs=opt.num_epochs,
                   lr=opt.lr, steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                   model_dir=root_path, loss_fn=loss_fn, iters_til_checkpoint=opt.iters_til_ckpt, summary_fn=summary_fn,
                   clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True, gpus=opt.gpus, rank=gpu)#,gpus=torch.cuda.device_count())

if __name__ == "__main__":
    opt = p.parse_args()

    if opt.gpus > 1:
        mp.spawn(main, nprocs=opt.gpus, args=(opt,))
    else:
        main(0, opt)

