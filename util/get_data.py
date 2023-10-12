from __future__ import print_function
import os
import tarfile
import requests
from warnings import warn
from zipfile import ZipFile
from bs4 import BeautifulSoup
from os.path import abspath, isdir, join, basename


class GetData(object):
    """

    Download CycleGAN or Pix2Pix Data.

    Args:
        technique : str
            One of: 'cyclegan' or 'pix2pix'.
        verbose : bool
            If True, print additional information.

    Examples:
        >>> from util.get_data import GetData
        >>> gd = GetData(technique='cyclegan')
        >>> new_data_path = gd.get(save_path='./datasets')  # options will be displayed.

    """

    def __init__(self, technique='cyclegan', verbose=True):
        url_dict = {
            'pix2pix': 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets',
            'cyclegan': 'https://people.eecs.berkeley.edu/~taesung_par