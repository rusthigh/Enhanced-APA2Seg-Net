
import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    if opt.dataset_mode == 'apada2seg_train':
        from data.apada2seg_dataset import TrainDataset
        dataset = TrainDataset()
    elif opt.dataset_mode == 'apada2seg_test':
        from data.apada2seg_test_dataset import TestDataset
        dataset = TestDataset()

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)