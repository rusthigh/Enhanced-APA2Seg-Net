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
        self.parser.add_argument('--raw_B_dir', type=str, default='./preprocess/MRI_SEG/PROC/MRI/'