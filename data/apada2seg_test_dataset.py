import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import torch
import random
import data.random_pair as random_pair


class TestDataset(BaseDataset):
    def initial