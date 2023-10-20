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
            'cyclegan': 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets'
        }
        self.url = url_dict.get(technique.lower())
        self._verbose = verbose

    def _print(self, text):
        if self._verbose:
            print(text)

    @staticmethod
    def _get_options(r):
        soup = BeautifulSoup(r.text, 'lxml')
        options = [h.text for h in soup.find_all('a', href=True)
                   if h.text.endswith(('.zip', 'tar.gz'))]
        return options

    def _present_options(self):
        r = requests.get(self.url)
        options = self._get_options(r)
        print('Options:\n')
        for i, o in enumerate(options):
            print("{0}: {1}".format(i, o))
        choice = input("\nPlease enter the number of the "
                       "dataset above you wish to download:")
        return options[int(choi