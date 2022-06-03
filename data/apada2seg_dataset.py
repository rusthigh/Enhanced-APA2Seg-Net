import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
import data.random_pair as random_pair
import pdb


class TrainDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        self.dir_A = opt.raw_A_dir
        self.dir_B = opt.raw_B_dir
        self.dir_Seg = opt.raw_A_seg_dir

        self.A_paths = opt.imglist_A
        self.B_paths = opt.imglist_B

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        if not self.opt.isTrain:
            self.skipcrop = True
            self.skiprotate = True
        else:
            self.skipcrop = False
            self.skiprotate = False

        if self.skipcrop:
            osize = [opt.fineSize, opt.fineSize]
        else:
            osize = [opt.loadSize, opt.loadSize]

        if self.skiprotate:
            