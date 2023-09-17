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
        for subline in sublines:
            sub_info = subline.replace('\n', '')
            sub_names.append(sub_info)
        fp.close()
        return sub_names
    else:
        fp = open(sub_list_file, 'w')
        img_root_dir = os.path.join(path)
        subs = os.listdir(img_root_dir)
        subs.sort()
        for sub in subs:
            sub_dir = os.path.join(img_root_dir,sub)
            views = os.listdir(sub