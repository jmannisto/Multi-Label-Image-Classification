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

# --- set up ---
alexnet = models.alexnet(weights = 'DEFAULT')
optimizer = optim.Adam(alexnet.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

# --- editing parameters of pretrained model ---
#taken from pytorch
#set up for AlexNet
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = models.alexnet(pretrained=use_pretrained) #pretrained = use final layer
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes) #change outputs to 14 
    input_size = 224 #image size needed
    return model_ft, input_size

#necessary if we only want to update the final layer
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model_ft, input_size = initialize_model(NUM_CLASSES, feature_extract=True, use_pretrained=True)

params_to_update = model_ft.parameters()
print("Params to learn:")
params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        #print("\t",name)
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# --- Dataset initialization ---
transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #these values were hardcoded for pretrained model
    ])

print('Preparing dataset')
#train_loader, dev_loader, test_loader = data_load.get_dataset(BATCH_SIZE_TRAIN)
train_loader, dev_loader, test_loader = data_load.get_dataset(BATCH_SIZE_TRAIN, transformations)

# --- eval functions ---
#TODO: edit accuracy method, currrently incorrect
def accuracy(output, gold):
    output = torch.sigmoid(output) #no sigmoid at the end of the model so implemented here to round to 1 or 0 next
    preds = torch.round(output)
    correct = (preds == gold).sum().item()
    acc = correct/gold.shape[0]
    return acc

def evaluate(model, loader, criterion):
    #compare inference time, model size(# of params), classification report, prec/recal
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch['image'])
            loss = criterion(pred, batch['label'].float())
            epoch_loss += loss.item()
            epoch_acc += accuracy(pred, batch['label'])
    return epoch_loss/len(loader), epoch_acc/len(loader)


# --- training ---
#taken from and modified: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html 
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
        test_epoch_acc = 0.0 #testing
        # Each epoch has a training and validation phase
            # Iterate over data.
        for batch in train_loader:
            inputs, labels = batch['image'], batch['label']

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_correct = accuracy(outputs, labels)
            #backprop
            loss.backward()
            optimizer.step()

            # statistics
            epoch_loss += loss.item()# * inputs.size(0) #is this last part needed? 
            epoch_acc += accuracy(outputs, labels)
            test_epoch_acc += train_correct #testing
            #print(epoch_loss, epoch_acc)

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss/len(train_loader), epoch_acc/len(train_loader)))
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion)
        val_loss_history.append(dev_loss)
        val_acc_history.append(dev_acc)
        print('Val Loss: {:.4f} Val Acc: {:.4f}'.format(dev_loss, dev_acc))
        
        # copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        #TODO: add early stopping
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

# --- run training --- 
model_ft, hist = train_model(model_ft, train_loader, dev_loader, criterion, optimizer_ft, num_epochs=10)
