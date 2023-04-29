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
        self.parser.add_argument('--output_nc_seg', type=int, default=