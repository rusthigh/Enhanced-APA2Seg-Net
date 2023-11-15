import random
import numpy as np
import torch
from torch.autograd import Variable


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
 