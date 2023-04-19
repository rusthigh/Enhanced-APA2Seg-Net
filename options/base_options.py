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

        self.parser.add_argument('--batchSize', t