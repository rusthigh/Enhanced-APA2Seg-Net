import os
import numpy as np
import h5py
import random
import linecache
import pdb


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dir2list(path, sub_list_file):
    if os.path.exists(sub_list_file):
        fp = open(sub_list_file, 'r')
        sublines = fp.readlines()
        sub_names = []
        for s