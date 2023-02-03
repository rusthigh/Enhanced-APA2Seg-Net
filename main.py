
import time
import os
import sublist
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


opt = TrainOptions().parse()

Method = opt.apada2seg_data_model

raw_A_dir = opt.raw_A_dir
raw_A_seg_dir = opt.raw_A_seg_dir
raw_B_dir = opt.raw_B_dir

TrainOrTest = opt.apada2seg_run_model   # 'Train' #

if TrainOrTest == 'Train':

    sub_list_A = opt.sub_list_A
    sub_list_B = opt.sub_list_B

    imglist_A = sublist.dir2list(raw_A_dir, sub_list_A)
    imglist_B = sublist.dir2list(raw_B_dir, sub_list_B)

    imglist_A, imglist_B = sublist.equal_length_two_list(imglist_A, imglist_B)

    # input the opt that we want
    opt.raw_A_dir = raw_A_dir
    opt.raw_A_seg_dir = raw_A_seg_dir
    opt.raw_B_dir = raw_B_dir
    opt.imglist_A = imglist_A
    opt.imglist_B = imglist_B

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print('#training images = %d' % dataset_size)
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    print('#model created')

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(), epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0: