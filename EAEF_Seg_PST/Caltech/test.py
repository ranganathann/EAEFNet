import os
import argparse
import time
import datetime
import stat
import shutil
import random
import warnings
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.caltech_dataset import caltech_dataset
from util.augmentation import RandomFlip, RandomCrop
from util.util import compute_results, visualize
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from pytorch_toolbelt import losses as L
from torch.cuda.amp import autocast, GradScaler
from model.EAEFNet import EAEFNet
import cv2
#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str,
                    default='EAEFNet50_hight_iou_2')
parser.add_argument('--batch_size', '-b', type=int, default=1)
parser.add_argument('--seed', default=3407, type=int,
                    help='seed for initializing training.')
parser.add_argument('--lr_start', '-ls', type=float, default=0.02)
parser.add_argument('--gpu', '-g', type=int, default=0)
#############################################################################################
parser.add_argument('--lr_decay', '-ld', type=float, default=0.95)
parser.add_argument('--epoch_max', '-em', type=int, default=60)
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=4)
parser.add_argument('--n_class', '-nc', type=int, default=10)
parser.add_argument('--loss_weight', '-lw', type=float, default=0.5)
parser.add_argument('--data_dir', '-dr', type=str,
                    default='D:/pst900/pst900_thermal_rgb-master/PST900_RGBT_Dataset/')
args = parser.parse_args()
#############################################################################################
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cuda")
scaler = GradScaler()


if __name__ == '__main__':
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')
    model = EAEFNet.EAEFNet(args.n_class)
    if args.gpu >= 0:
        model.cuda(args.gpu)

    weight_dir = os.path.join("./runs/", args.model_name)
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir, mode=0o777)
    # os.chmod(weight_dir,
    #          stat.S_IRWXO)  # allow the folder created by docker read, written, and execuated by local machine

    writer = SummaryWriter("./runs/tensorboard_log")
    # allow the folder created by docker read, written, and execuated by local machine
    # os.chmod("./runs/tensorboard_log", stat.S_IRWXO)
    # os.chmod("./runs", stat.S_IRWXO)

    train_dataset = caltech_dataset(
        data_dir=args.data_dir, split='train')
    val_dataset = caltech_dataset(data_dir=args.data_dir, split='val')
    test_dataset = caltech_dataset(
        data_dir=args.data_dir, split='test')

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    model.load_state_dict(torch.load(os.path.join(weight_dir, "59.pth")))
    model.eval()
    with torch.no_grad():
        for it, (images, thermal, labels, names) in enumerate(test_loader):
            if it > 50:
                break
            images = Variable(images).cuda(args.gpu)
            thermal = Variable(thermal).cuda(args.gpu)
            labels = Variable(labels).cuda(args.gpu)
            images = torch.cat([images, thermal], dim=1)
            start_t = time.time()  # time.time() returns the current time

            logits_S, logits_T = model(images)
            scale = max(1, 255 // args.n_class)
            # logit_mix = (logit + logits)
            # predicted_tensor = logits_T.argmax(1).unsqueeze(1) * scale
            # predicted_tensor = torch.cat(
            #             (predicted_tensor, predicted_tensor, predicted_tensor), 1)
            predicted_tensor = logits_T.argmax(1).unsqueeze(1)
            # prediction = logit_mix.argmax(1).cpu().numpy().squeeze().flatten()
            pred_img = predicted_tensor[0, :, :, :].cpu().numpy()
            cv2.imwrite(os.path.join(
                "./results", names[0]), np.swapaxes(np.swapaxes(pred_img, 0, 2), 1, 0))
