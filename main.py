

import torch
import torchvision
from syed_eva8.utils import data_handling, train, test, gradcam, helpers, augmentation
from syed_eva8.models import resnet18
from torch.optim.lr_scheduler import StepLR, ExponentialLR, OneCycleLR, LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


def create_dataloaders(mean, std, cuda, config, augment_func = "albumentation_augmentation", gpu_batch_size = 512, cpu_batch_size = 64):
    
    ## Define data transformations
    train_transforms, test_transforms = eval("augmentation."+augment_func+"(mean, std, config)")

    ## Download & return transformed datasets
    trainset, testset = data_handling.return_datasets(train_transforms, test_transforms)

    ## Define data loaders
    trainloader, testloader = data_handling.return_dataloaders(trainset, testset, cuda, gpu_batch_size, cpu_batch_size)
    
    return trainloader, testloader


def trigger_training(model, device, trainloader, testloader, config, optimizer_name = "Adam", scheduler_name = "OneCycle", criterion_name = "CrossEntropyLoss", lambda_l1 = 0, epochs = 100):
    
    train_acc, train_losses, test_acc, test_losses, lrs = [], [], [], [], []
    
    if (optimizer_name == "Adam"):
        optimizer = optim.Adam(model.parameters(), lr = config['standard_lr'], weight_decay = config['L2_penalty'])
    else:
        optimizer = optim.SGD(model.parameters(), lr = config['standard_lr'], momentum = config['momentum_val'])
        
    if (scheduler_name == "OneCycle"):
        scheduler = OneCycleLR(optimizer, max_lr = config['standard_lr'], epochs = epochs, steps_per_epoch = len(trainloader), pct_start = config['oneCycle_pct_start'])
    elif (scheduler_name == "ReduceLROnPlateau"):
        scheduler = ReduceLROnPlateau(optimizer, mode = config['sch_reduceLR_mode'], factor = config['sch_reduceLR_factor'], patience = config['sch_reduceLR_patience'], threshold = config['sch_reduceLR_threshold'], threshold_mode = config['sch_reduceLR_threshold_mode'], cooldown = config['sch_reduceLR_cooldown'], min_lr = config['sch_reduceLR_min_lr'], eps = config['sch_reduceLR_eps'], verbose = False)
    elif (scheduler_name == "None"):
        scheduler = "None"
        
    if (criterion_name == "CrossEntropyLoss"):
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = F.nll_loss
     
    for epoch in range(epochs):
        print("EPOCH:", epoch + 1)
        train.train(model, device, trainloader, train_acc, train_losses, optimizer, scheduler, criterion, lrs, lambda_l1)
        test.test(model, device, testloader, test_acc, test_losses, scheduler, criterion)
    
    
    return train_acc, train_losses, test_acc, test_losses, lrs
