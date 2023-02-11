## Helper functions
import torch
import matplotlib.pyplot as plt
import yaml
import numpy as np

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## Check for GPU and set device accordingly
def gpu_check(seed_value = 1):
    
    ##Set seed for reproducibility
    SEED = seed_value

    # CUDA?
    cuda = torch.cuda.is_available()
    if cuda:
        print("CUDA is available")
    else:
        print("CUDA unavailable")

    # For reproducibility
    torch.manual_seed(SEED)

    if cuda:
        torch.cuda.manual_seed(SEED)
    
    device = torch.device("cuda" if cuda else "cpu")
    
    return device, cuda

def plot_metrics(train_acc, train_losses, test_acc, test_losses):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    
def accuracy_per_class(classes, testloader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        
def wrong_predictions(model, test_loader, device):
    wrong_images=[]
    wrong_label=[]
    correct_label=[]
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)        
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])
            wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))
        print(f'Total wrong predictions are {len(wrong_predictions)}')

    return wrong_predictions

def plot_misclassified(wrong_predictions, mean, std, num_img):
    fig = plt.figure(figsize=(15,12))
    fig.tight_layout()
    for i, (img, pred, correct) in enumerate(wrong_predictions[:num_img]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j]*std[j])+mean[j]

        img = np.transpose(img, (1, 2, 0)) 
        ax = fig.add_subplot(5, 5, i+1)
        fig.subplots_adjust(hspace=.5)
        ax.axis('off')
        #class_names,_ = get_classes()

        ax.set_title(f'\nActual : {classes[target.item()]}\nPredicted : {classes[pred.item()]}',fontsize=10)  
        ax.imshow(img)  

    plt.show()
    
def load_config_variables(file_name):
    with open(file_name, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            print(" Loading config ..")
            #globals().update(config)
            print(" Config succesfully loaded ")
            return config
        except ValueError:
            print("Invalid yaml file")
            exit(-1)