import argparse
import os
from pyexpat import model
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import ImageNet

import sys
sys.path.append('../pytorch-classification-master/models/cifar')
import resnet
import numpy as np


DATA_DIR = '/n/fs/ptml/dingliy/saliency/adversarial/data'
OUT_DIR = '/n/fs/ptml/arushig/saliency/adversarial/data'
# ImageNet stuff
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]    
IMAGENET_DIR = '/n/fs/ptml/datasets/imagenet/'
# Cifar-10 stuff
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]
CIFAR10_DIR = '/n/fs/ptml/datasets/cifar10/'
CIFAR10_MODEL_DIR = '/n/fs/ptml/nsaunshi/saliency/pytorch-classification-master/models/cifar/cifar10/'
CIFAR100_DIR = '/n/fs/ptml/datasets/cifar100/'
CIFAR100_MODEL_DIR = '/n/fs/ptml/nsaunshi/saliency/pytorch-classification-master/models/cifar/cifar100/'



def completeness(auc_probs, probs, epsilon1=0.01, alpha=1.0):
    return np.minimum(np.maximum(epsilon1, auc_probs) / np.minimum(alpha, probs), 1)

def soundness(auc_probs, probs, beta=0.0):
    return np.minimum(np.maximum(beta, probs) / auc_probs, 1)

def imagenet_loaders(data_dir=IMAGENET_DIR):
    mean = IMAGENET_MEAN
    std = IMAGENET_STD
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    dataset = ImageNet(root=data_dir, split='train', transform=transform_train)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    dataset = ImageNet(root=data_dir, split='val', transform=transform_val)
    val_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    return train_loader, val_loader


def cifar10_loaders(data_dir=CIFAR10_DIR):
    mean = CIFAR_MEAN
    std = CIFAR_STD
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataloader = datasets.CIFAR10
    num_classes = 10

    trainset = dataloader(root='/n/fs/ptml/datasets/cifar10', train=True, download=True, transform=transform_train)
    tr_loader = DataLoader(trainset, batch_size=128, shuffle=True)

    testset = dataloader(root='/n/fs/ptml/datasets/cifar10', train=False, download=False, transform=transform_test)
    te_loader = DataLoader(testset, batch_size=128, shuffle=True)
    return tr_loader, te_loader



def cifar10_resnet164(data_dir=CIFAR10_MODEL_DIR):
    model = resnet.resnet(depth=164, num_classes=10, block_name='bottleneck').cuda()
    checkpoint_pth = os.path.join(CIFAR10_MODEL_DIR, 'resnet-110', 'model_best.pth.tar')
    print("=> loading checkpoint '{}'".format(checkpoint_pth))
    checkpoint = torch.load(checkpoint_pth)
    state_dict = checkpoint['state_dict']
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(checkpoint['acc'], checkpoint['best_acc'])
    model.eval()
    return model



def cifar100_loaders(data_dir=CIFAR100_DIR):
    mean = CIFAR_MEAN
    std = CIFAR_STD
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataloader = datasets.CIFAR100
    num_classes = 10

    trainset = dataloader(root='/n/fs/ptml/datasets/cifar100', train=True, download=True, transform=transform_train)
    tr_loader = DataLoader(trainset, batch_size=128, shuffle=True)

    testset = dataloader(root='/n/fs/ptml/datasets/cifar100', train=False, download=False, transform=transform_test)
    te_loader = DataLoader(testset, batch_size=128, shuffle=True)
    return tr_loader, te_loader



def cifar100_resnet164(data_dir=CIFAR100_MODEL_DIR):
    model = resnet.resnet(depth=164, num_classes=100, block_name='bottleneck').cuda()
    checkpoint_pth = os.path.join(CIFAR100_MODEL_DIR, 'resnet-110', 'model_best.pth.tar')
    print("=> loading checkpoint '{}'".format(checkpoint_pth))
    checkpoint = torch.load(checkpoint_pth)
    state_dict = checkpoint['state_dict']
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(checkpoint['acc'], checkpoint['best_acc'])
    model.eval()
    return model

class ReducedModel(torch.nn.Module):
    def __init__(self, model, label_list):
        super(ReducedModel, self).__init__()
        self.model = model
        self.label_list = label_list
    
    def forward(self, input):
        return self.model(input)[..., self.label_list]

def parse():

    parser = argparse.ArgumentParser(description='Learning certificates for saliency')
    parser.add_argument('--steps', default=2000, type=int,
                        help='number of total steps to run per images')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate')
    parser.add_argument('--bs', default=20, type=int,
                        help='mini-batch size for number of images to be processed together')
    parser.add_argument('--noise_mode', default='rand_image', type=str,
                        help='type of noise model for remaining pixels (gray, rand_image)')
    parser.add_argument('--noise_bs', default=5, type=int,
                        help='mini-batch size for noise images')
    parser.add_argument('--K', default=1, type=int,
                        help='number of certificates to learn per image')




    parser.add_argument('--label_list', default=[], type=int, nargs='*',
                        help='list of labels to run for ')
    parser.add_argument('--scale_list', default=[1, 4], type=int, nargs='*',
                        help='upsampling factor for mask')
    parser.add_argument('--reg_l1_list', default=[0.00002], type=float, nargs='*',
                        help='regularization coefficient for l1 norm of masks')
    parser.add_argument('--reg_tv_list', default=[0., 0.01], type=float, nargs='*',
                        help='regularization coefficient for TV of masks')
    parser.add_argument('--reg_ent_list', default=[0.0], type=float, nargs='*',
                        help='regularization coefficient for entropy of across K masks')
    parser.add_argument('--obj', default='xent', type=str,
                        help='objective function for mask')
    
    parser.add_argument('--fit_label', default='correct', type=str,
                        help='which label to find saliency map for (correct, sbest, rand, 1, 2, 3, ..)')

    parser.add_argument('--pgd_steps', default=20, type=int,
                        help='number of total PGD steps for adversary to run per images')
    parser.add_argument('--eps', default=0.031372549019608, type=float,
                        help='linf eps')


    parser.add_argument('--imagenet_dir', default=IMAGENET_DIR, type=str,
                        help='ImageNet data directory')
    parser.add_argument('--label_input_path', default='input_data/sample_labels.npy', type=str,
                        help='file path to labels')
    parser.add_argument('--out_dir', default=OUT_DIR, type=str,
                        help='Directory to store output masks')
    parser.add_argument('--image_input_path', default='input_data/sample_images.npy', type=str,
                        help='file path to images')
    parser.add_argument('--start_end', default=[0, 1556], type=int, nargs=2,
                        help='start and end indices of images to process (for manual parallelization)')
    parser.add_argument('--curve_start_end', default=[0, 1], type=float, nargs=2,
                        help='start and end indices of AUC curve')
    parser.add_argument('--start_point', default=[0, 0], type=int, nargs=2,
                        help='start idx and label for generating maps')
    

    parser.add_argument('--debug', action='store_true',
                        help='print intermediate outputs or not')


    parser.set_defaults(augment=False)

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, " : ", getattr(args, arg))

    return args
