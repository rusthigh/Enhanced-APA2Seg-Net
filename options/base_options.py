import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--raw_A_dir', type=str, default='./preprocess/MRI_SEG/PROC/DCT/', help='data dir to domain A')
        self.parser.add_argument('--raw_A_seg_dir', type=str, default='./preprocess/MRI_SEG/PROC/DCT/', help='data dir to domain A segmentation')
        self.parser.add_argument('--raw_B_dir', type=str, default='./preprocess/MRI_SEG/PROC/MRI/', help='data dir to domain B')
        self.parser.add_argument('--sub_list_A', type=str, default='./preprocess/MRI_SEG/PROC/train_DCT.txt', help='list file for domain A')
        self.parser.add_argument('--sub_list_B', type=str, default='./preprocess/MRI_SEG/PROC/train_MRI.txt', help='list file for domain B')

        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--angle', type=int, default=10, help='random rotation angle [-angle angle]')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--input_nc_seg', type=int, default=1, help='# of input image channels for segmentation')
        self.parser.add_argument('--output_nc_seg', type=int, default=2, help='# of output image channels for segmentation')
        self.parser.add_argument('--seg_norm', type=str, default='DiceNorm', help='DiceNorm or CrossEntropy')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG: resnet_9blocks/resnet_6blocks/unet/duseunet')
        self.parser.add_argument('--which_model_netS', type=str, default='duseunet', help='selects model to use for netS: resnet_9blocks/resnet_6blocks/unet/duseunet')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='apada2seg_train', help='chooses how datasets are loaded. [unaligned | aligned | single | apada2seg]')
        self.parser.add_argument('--model', type=str, default='apada2seg_model_train', help='chooses which model to use. cycle