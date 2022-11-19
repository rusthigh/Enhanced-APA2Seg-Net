import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
import data.random_pair as random_pair


class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt

        self.dir_B = opt.test_B_dir

        self.B_filenames = opt.imglist_testB
        self.B_size = len(self.B_filenames)

        if not self.opt.isTrain:
            self.skipcrop = True
            self.skiprotate = True
        else:
            self.skipcrop = False
            self.skiprotate = False

        if self.skipcrop:
            osize = [opt.fineSize, opt.fineS