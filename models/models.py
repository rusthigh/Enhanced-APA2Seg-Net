
def create_model(opt):
    print(opt.model)

    # our anatomy-preserving adaptation segmentation
    if opt.model == 'apada2seg_model_train':
        assert(opt.dataset_mode == 'apada2seg_train')
        from .apada2seg_model import 