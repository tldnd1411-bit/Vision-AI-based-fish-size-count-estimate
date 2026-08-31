import torch.utils.data
import torchvision

from .SHA import build as build_sha
from .Oliver_Flounder import build as build_Oliver_Flounder

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
    'Oliver_Flounder': './data/Oliver_Flounder/',
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'SHA':
        return build_sha(image_set, args)
    if args.dataset_file == 'Oliver_Flounder':
        return build_Oliver_Flounder(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')