# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 14:52:54 2023

@author: syeda
"""

from torchvision import datasets, transforms
import torchvision
import torch
import numpy as np
import os
from Torch_CV_Utils.utils import augmentation


cifar10_classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## Calculate Dataset Statistics
def return_dataset_statistics():
    
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root='./data', train = True, download = True, transform = train_transform)
    
    mean = train_set.data.mean(axis=(0,1,2))/255
    std = train_set.data.std(axis=(0,1,2))/255

    return mean, std


def return_datasets(train_transforms, test_transforms):
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform = train_transforms)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform = test_transforms)
    
    return trainset, testset

def return_dataloaders(trainset, testset, cuda, gpu_batch_size = 128, cpu_batch_size = 64):
    
    dataloader_args = dict(shuffle = True, batch_size = gpu_batch_size, num_workers = 4, pin_memory = True) if cuda else dict(shuffle = True, batch_size = cpu_batch_size)

    trainloader = torch.utils.data.DataLoader(trainset, **dataloader_args)

    testloader = torch.utils.data.DataLoader(testset, **dataloader_args)
    
    return trainloader, testloader

def prep_tinyimagenet(valid_dir):
    
    val_img_dir = os.path.join(valid_dir, 'images')

    # Open and read val annotations text file
    fp = open(os.path.join(valid_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()

    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))
    
    return val_img_dir

def create_tinyimagenet_datasets(train_dir, val_img_dir, mean, std, config, augment_func):
    
    ## Define data transformations
    train_transforms, test_transforms = eval("augmentation."+augment_func+"(mean, std, config)")

    trainset = datasets.ImageFolder(train_dir, transform = train_transforms)
    testset = datasets.ImageFolder(val_img_dir, transform = test_transforms)

    return trainset, testset

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    
    
## Function to show sample data
def show_sample_data(trainloader, num_images = 16):
    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images[0:num_images]))
    # print labels
    print(' '.join('%5s' % cifar10_classes[labels[j]] for j in range(num_images)))