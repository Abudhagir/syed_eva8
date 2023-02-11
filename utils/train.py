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
