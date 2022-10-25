import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import config as config
from models import Decoder
from dataloaders.dataloader import MyDataloader
import argparse

cmap = plt.cm.viridis


def parse_command():
    data_names = ['kitti']
    model_names = ['MobileNetV3SkipAddL_NNConv5R', 'MobileNetV3SkipAddL_NNConv5S', 'MobileNetV3L_NNConv5GU', 'MobileNetV3SkipAddS_NNConv5R', 'MobileNetV3S_NNConv5GU']
    loss_names = ['l1', 'l2']
    decoder_names = Decoder.names
    modality_names = MyDataloader.modality_names

    parser = argparse.ArgumentParser(description='FastDepth')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: MobileNetV3SkipAddS_NNConv5R)')
    parser.add_argument('--data', metavar='DATA', default='kitti',
                        choices=data_names,
                        help='dataset: ' + ' | '.join(data_names) + ' (default: kitti)')
    parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                        help='number of sparse depth samples (default: 0)')
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=modality_names,
                        help='modality: ' + ' | '.join(modality_names) + ' (default: rgb)')
    parser.add_argument('--max-depth', default=-1.0, type=float, metavar='D',
                        help='cut-off depth of sparsifier, negative values means infinity (default: inf [m])')
    parser.add_argument('--decoder', '-d', metavar='DECODER', default='nnconv', choices=decoder_names,
                        help='decoder: ' + ' | '.join(decoder_names) + ' (default: nnconv)')
    parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run (default: 15)')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_names,
                        help='loss function: ' + ' | '.join(loss_names) + ' (default: l1)')
    parser.add_argument('-b', '--batch-size', default=8, type=int, help='mini-batch size (default: 8)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='',
                        type=str, metavar='PATH',
                        help='path to latest checkpoint (default: results/kitti.samples=0.modality=rgb.arch=MobileNetV3SkipAdd.decoder=nnconv.criterion=l1.lr=0.01.bs=8.pretrained=True/ch.peckpoint-4.pth.tar)')
    parser.add_argument('-e', '--evaluate', dest = 'evaluate', type = str, default='')
    parser.add_argument('-t', '--train', default='', type=str)
    parser.add_argument('-pt', '--no-pretrain', dest='pretrained', action='store_false',
                        help='not to use ImageNet pre-trained weights')

    parser.set_defaults(pretrained=True)
    if config.GPU == True:
        parser.set_defaults(cuda=True)
    args = parser.parse_args()

    if args.modality == 'rgb' and args.num_samples != 0:
        print("number of samples is forced to be 0 when input modality is rgb")
        args.num_samples = 0
    if args.modality == 'rgb' and args.max_depth != 0.0:
        print("max depth is forced to be 0.0 when input modality is rgb/rgbd")
        args.max_depth = 0.0
    return args

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def adjust_learning_rate(optimizer, epoch, lr_init):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_output_directory(args):
    print("Inside output directory method")
    output_directory = os.path.join('results',
        '{}.samples={}.modality={}.arch={}.decoder={}.criterion={}.lr={}.bs={}.pretrained={}'.
        format(args.data,  args.num_samples, args.modality, \
            args.arch, args.decoder, args.criterion, args.lr, args.batch_size, \
            args.pretrained))
    print("Output directory ",output_directory)
    return output_directory

def get_depth_map(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    return depth_target_col,depth_pred_col
