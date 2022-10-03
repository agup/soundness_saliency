import numpy as np
import torch
import os
import pickle as pkl
import numpy as np
import torchvision
import random
from certificate_methods import *
from utils import parse, cifar10_resnet164, cifar10_loaders
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageNet
import os
os.environ['TORCH_HOME'] = './'

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ImageNet stuff

IMAGENET_DIR = '/n/fs/ptml/datasets/imagenet/'

transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),])


transform_test = transforms.Compose([transforms.Resize(256),
                transforms.CenterCrop(224),transforms.ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]) ])

label_list = [526, 527, 664, 673, 782, 851]
class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path



imagenet_data = torchvision.datasets.ImageNet(IMAGENET_DIR,  split='train', transform = transform_train)
imagenet_data_val = torchvision.datasets.ImageNet(IMAGENET_DIR,  split='val',   transform = transform_test) 

#pick some sample input images from the test set 
val_loader =  torch.utils.data.DataLoader(imagenet_data_val, batch_size = 200, shuffle = True)

for i, (b_x, b_y) in enumerate(val_loader):
    if i ==0 :
        relevant = np.isin(b_y.cpu().numpy(), label_list)
        batch_x = b_x[relevant].cpu().numpy()
        batch_y = b_y[relevant].cpu().numpy()
    elif i > 0 and len(batch_x) < 100:
        relevant = np.isin(b_y, label_list)
        batch_x = np.concatenate([batch_x,b_x[relevant].cpu().numpy()], 0 )
        batch_y = np.concatenate([batch_y,b_y[relevant].cpu().numpy()], 0 )
    else:
        break
np.save('./input_data/sample_images.npy', batch_x)
np.save('./input_data/sample_labels.npy', batch_y)


