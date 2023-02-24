import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt

## Get Learning Rate
def get_lr(optimizer):
    
    for param_group in optimizer.param_groups:
        return param_group['lr']

def lr_range_test(model, train_loader, test_loader, device, criterion, start_lr = 1e-5, end_lr=10, mult_factor = 1.3):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    lambda1 = lambda epoch: mult_factor ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    model.train()
    pbar = tqdm(train_loader)
    train_losses, test_losses, lrs = [],[], []

    for batch_idx, (data, target) in enumerate(pbar):
        
        ## Get data samples
        data, target = data.to(device), target.to(device)

        ## Init
        optimizer.zero_grad()

        ## Predict
        y_pred = model(data)

        ## Calculate loss
        loss = criterion(y_pred, target)

        train_losses.append(loss.data.cpu().numpy().item())

        ## Backpropagation
        loss.backward()

        optimizer.step()
        scheduler.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        current_lr = get_lr(optimizer)
        lrs.append(current_lr)
        if (current_lr > 10):
            break

    return test_losses, lrs


def plot_lrs(lrs, test_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.plot(lrs)
    ax1.set_title("Learning Rate")
    ax2.plot(lrs, test_losses)
    ax2.set_xscale('log')
    ax2.set_title("Learning rate vs Loss")