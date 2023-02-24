import numpy as np
import torch
from tqdm import tqdm


def train(model, device, train_loader, criterion, scheduler, optimizer, use_l1=False, lambda_l1=0.01):
    """Function to train the model

    Args:
        model (instance): torch model instance of defined model
        device (str): "cpu" or "cuda" device to be used
        train_loader (instance): Torch Dataloader instance for trainingset
        criterion (instance): criterion to used for calculating the loss
        scheduler (function): scheduler to be used
        optimizer (function): optimizer to be used
        use_l1 (bool, optional): L1 Regularization method set True to use . Defaults to False.
        lambda_l1 (float, optional): Regularization parameter of L1. Defaults to 0.01.

    Returns:
        float: accuracy and loss values
    """
    model.train()
    pbar = tqdm(train_loader)
    lr_trend = []
    correct = 0
    processed = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch 
        # accumulates the gradients on subsequent backward passes. Because of this, when you start your training loop, 
        # ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)
        # Calculate loss
        loss = criterion(y_pred, target)

        l1=0
        if use_l1:
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
        loss = loss + lambda_l1*l1

        # Backpropagation
        loss.backward()
        optimizer.step()
        # updating LR
        if scheduler:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                lr_trend.append(scheduler.get_last_lr()[0])

        train_loss += loss.item()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        

        pbar.set_description(desc= f'Batch_id={batch_idx} Loss={train_loss/(batch_idx + 1):.5f} Accuracy={100*correct/processed:0.2f}%')
    return 100*correct/processed, train_loss/(batch_idx + 1), lr_trend

'''
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.contrib.handlers import FastaiLRFinder, ProgressBar

def range_test(model, optimizer, criterion, device, trainloader, testloader, start_lr = 1e-4, end_lr = 10, max_epochs = 10):
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: {"batch loss": x})

    lr_finder = FastaiLRFinder()
    to_save={'model': model, 'optimizer': optimizer}
    with lr_finder.attach(trainer, to_save, start_lr = start_lr, end_lr = end_lr, diverge_th=1.5) as trainer_with_lr_finder:
        trainer_with_lr_finder.run(trainloader)
        
    trainer.run(trainloader, max_epochs=10)

    evaluator = create_supervised_evaluator(model, metrics={"acc": Accuracy(), "loss": Loss(criterion)}, device=device)
    evaluator.run(testloader)

    print(evaluator.state.metrics)

    return lr_finder



import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


## Get Learning Rate
def get_lr(optimizer):
    
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, device, train_loader, train_acc, train_loss, optimizer, scheduler, criterion, lrs, lambda_l1 = 0, grad_clip = None):

    model.train()
    pbar = tqdm(train_loader)
  
    correct = 0
    processed = 0
  
    for batch_idx, (data, target) in enumerate(pbar):
        
        ## Get data samples
        data, target = data.to(device), target.to(device)

        ## Init
        optimizer.zero_grad()

        ## Predict
        y_pred = model(data)

        ## Calculate loss
        loss = criterion(y_pred, target)

        ## L1 Regularization
        if lambda_l1 > 0:
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + lambda_l1*l1

        train_loss.append(loss.data.cpu().numpy().item())

        ## Backpropagation
        loss.backward()

        # Gradient clipping
        if grad_clip: 
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()
        if ("ReduceLROnPlateau" in str(scheduler)):
            pass ##scheduler.step() will be updated in the test function for this scheduler option
        elif ("None" in str(scheduler)):
            print("Skipping scheduler step")
        else:    
            scheduler.step()
        
        lrs.append(get_lr(optimizer))

        ## Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} LR={lrs[-1]:0.5f} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)
'''