import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import time
import copy
import sys
import os
from sklearn.metrics import classification_report
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) #having issues with file path in VSCode
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append('../data')
from data import data_load

# --- hyperparameters ---
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
LR = 0.001

# --- fixed constants ---
NUM_CLASSES = 14
#TODO: probably edit this directory file path
DATA_DIR = '../../data/interim/'
TRAIN_TEST_DEV_RATIO = [0.8, 0.1, 0.1] #TODO: pass this through to data load

#--- set up device ---
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# --- set up ---
alexnet = models.alexnet(weights = 'DEFAULT')
optimizer = optim.Adam(alexnet.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

# --- editing parameters of pretrained model ---
#taken from pytorch: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html 

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    Initialize model based on model name passed, each variable is model specific
    """
    if model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224 
    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

#necessary if we only want to update the final layer
#also from pytorch: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html 
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# --- Dataset initialization ---
transformations = transforms.Compose([
        #transforms.ColorJitter()
        #transforms.RandomRotation(25) #added occasionally for testing
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #these values were hardcoded for pretrained model
    ])

print('Preparing dataset')
train_loader, dev_loader, test_loader = data_load.get_dataset(BATCH_SIZE_TRAIN, transformations)

# --- eval functions ---
def accuracy(output, gold):
    total = 0
    correct = 0 
    output = torch.sigmoid(output)
    preds = torch.round(output)
    total += gold.size(0)
    correct += (preds == gold).sum().item()
    acc = correct/(total*NUM_CLASSES)
    return acc

def evaluate(model, loader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            pred = model(inputs)
            loss = criterion(pred, labels)
            epoch_loss += loss.item() 
            epoch_acc += accuracy(pred,labels)
    return epoch_loss/len(loader), epoch_acc/len(loader)

def testing(model, test_loader, criterion):
    all_predicted = np.zeros((1, NUM_CLASSES))
    all_labels = np.zeros((1, NUM_CLASSES))
    evaluation_loss = 0.0
    evaluation_corrects = 0
    evaluation_total = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            evaluation_loss += loss.item()
            predicted = np.round(torch.sigmoid(outputs).cpu()).cpu()
            all_predicted = np.append(all_predicted, predicted, axis=0)
            all_labels = np.append(all_labels, labels.cpu(), axis=0)
            evaluation_corrects += (predicted.cpu() == labels.cpu()).sum().item()
            evaluation_total += labels.size(0)
        report = classification_report(all_labels[1:], all_predicted[1:])
        print(report)
    return evaluation_loss/len(test_loader.dataset), evaluation_corrects/(evaluation_total*NUM_CLASSES)*100

# --- training ---
#modified from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html 
def train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=10):
    since = time.time()
    val_acc_history = []
    val_loss_history = [] #for early stopping

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        epoch_loss = 0.0
        epoch_acc = 0.0
        
        # Iterate over data.
        for batch in train_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #backprop
            loss.backward()
            optimizer.step()

            # statistics
            epoch_loss += loss.item()
            epoch_acc += accuracy(outputs, labels)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss/len(train_loader), epoch_acc/len(train_loader)))
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion)
        val_loss_history.append(dev_loss)
        val_acc_history.append(dev_acc)
        print('Val Loss: {:.4f} Val Acc: {:.4f}'.format(dev_loss, dev_acc))
        
        # copy the model
        if (epoch_acc/len(train_loader)) > best_acc:
            best_acc = epoch_acc/len(train_loader)
            best_model_wts = copy.deepcopy(model.state_dict())
            
        #TODO: add early stopping

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Val Acc: {:4f}'.format(best_acc))

    #load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# --- run training --- 
#initialize, feature_extract = True means only final layer is updated while False means full finetune 
model_ft, input_size = initialize_model("densenet", NUM_CLASSES, feature_extract=True, use_pretrained=True) 

#also from pytorch: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html 
params_to_update = model_ft.parameters()
print("Params to learn:")
params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
optimizer_ft = optim.SGD(params_to_update, lr=0.001)

model_ft, hist = train_model(model_ft, train_loader, dev_loader, criterion, optimizer_ft, num_epochs=10)
