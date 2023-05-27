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

        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--angle', type=int, default=10, help='random rotation angle [-angle angle]')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels')
        self.parser.add_argument('--input_nc_seg', type=int, default=1, help='# of input image channels for segmentation')
        self.parser.add_argument('--output_nc_seg', type=int, default=2, help='# of output image channels for segmentation')
        self.parser.add_argument('--seg_norm', type=str, default='DiceNorm', help='DiceNorm or CrossEntropy')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG: resnet_9blocks/resnet_6blocks/unet/duseunet')
        self.parser.add_argument('--which_model_netS', type=str, default='duseunet', help='selects model to use for netS: resnet_9blocks/resnet_6blocks/unet/duseunet')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='apada2seg_train', help='chooses how datasets are loaded. [unaligned | aligned | single | apada2seg]')
        self.parser.add_argument('--model', type=str, default='apada2seg_model_train', help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./Checkpoints/', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=0, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--apada2seg_run_model', type=str, default='Train', help='choose: Train, TestSeg')
        self.parser.add_argument('--apada2seg_data_model', type=str, default='ImageWithMask', help='chooses the data location')

        self.parser.add_argument('--test_B_dir', type=str, default='./preprocess/MRI_SEG/PROC/MRI/', help='for test seg')
        self.parser.add_argument('--test_img_list_file', type=str, default='./preprocess/MRI_SEG/PROC/MRI/test_MRI.txt', help='test image list file')
        self.parser.add_argument('--test_seg_output_dir', type=str, default='./Output/MRI/experiment_apada2seg/', help='save test segmentation output results')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
