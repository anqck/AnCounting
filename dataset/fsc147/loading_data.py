from .fsc147 import FSC147
from .utils import resizeImage, resizeImageWithGT, resizeImageWithGT_HW

import torchvision.transforms as standard_transforms
from torchvision import transforms
import torch

MIN_HW = 384
MAX_HW = 1584

# DeNormalize used to get original images
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def collate_fn_fsc147(batch):
    # re-organize the batch
    batch_new = []
    for b in batch:
        imgs, examplers, density_gt, points = b
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        for i in range(len(imgs)):
            batch_new.append((imgs[i, :, :, :], examplers[i,0:3], density_gt[i],points[i]))
    batch = batch_new
    batch = list(zip(*batch))
    # batch[0] = nested_tensor_from_tensor_list(batch[0])

    batch[0] = torch.stack(batch[0])
    batch[1] = torch.stack(batch[1])
    batch[2] = torch.stack(batch[2])
    return tuple(batch)
    
def loading_data(data_path):
    # the pre-proccssing transform
    Transform = transforms.Compose([resizeImage( MAX_HW)])


    TransformTrain = transforms.Compose([resizeImageWithGT_HW(384,384)])
    # TransformTrain = transforms.Compose([resizeImageWithGT(MAX_HW)])

    # create the training dataset
    train_set = FSC147(data_path, transform=TransformTrain, train='train')
    # create the validation dataset
    val_set = FSC147(data_path, transform=TransformTrain, train='val')
    # create the test dataset
    test_set = FSC147(data_path, transform=TransformTrain, train='test')

    return train_set, val_set, test_set