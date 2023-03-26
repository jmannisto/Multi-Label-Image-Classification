import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from sklearn import metrics
from torchvision import transforms, datasets 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import time
import copy
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append('../data')
from data import data_load

N_EPOCHS = 10
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
LR = 0.001
MOMENTUM = 0.9
NUM_CLASSES = 14

def initialize_model(feature_extract, use_pretrained=True):
    model_ft = models.vgg11_bn(pretrained=use_pretrained) 
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[6] = nn.Linear(4096, NUM_CLASSES) 
    input_size = 224 
    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model_ft, input_size = initialize_model(feature_extract=True, use_pretrained=True)
optimizer = optim.SGD(model_ft.parameters(), lr=LR, momentum=MOMENTUM)
criterion = nn.BCEWithLogitsLoss()

params_to_update = model_ft.parameters()
params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
optimizer_ft = optim.SGD(params_to_update, lr=LR)

transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

print('Preparing dataset')
train_loader, dev_loader, test_loader = data_load.get_dataset(BATCH_SIZE_TRAIN, transformations)

def evaluate(model, loader, criterion):
    all_predicted = np.zeros((1, NUM_CLASSES))
    all_labels = np.zeros((1, NUM_CLASSES))
    evaluation_loss = 0.0
    evaluation_corrects = 0
    evaluation_total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['image'], batch['label']
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            evaluation_loss += loss.item() * inputs.size(0)
            predicted = np.round(torch.sigmoid(outputs).detach()).int()
            all_predicted = np.append(all_predicted, predicted, axis=0)
            all_labels = np.append(all_labels, labels, axis=0)
            evaluation_corrects += (predicted == labels).sum().item()
            evaluation_total += labels.size(0)
            report = metrics.classification_report(all_labels[1:], all_predicted[1:])
    return evaluation_loss/len(loader.dataset), evaluation_corrects/(evaluation_total*NUM_CLASSES)*100, report


def train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=10):
    start = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        for batch in train_loader:
            inputs, labels = batch['image'], batch['label']

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predicted = np.round(torch.sigmoid(outputs).detach())
            running_corrects += (predicted == labels).sum().item()
            running_total += labels.size(0)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss/len(train_loader.dataset)
        epoch_acc = running_corrects/(running_total*NUM_CLASSES)*100

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        dev_loss, dev_acc, dev_report = evaluate(model, dev_loader, criterion)
        print(dev_report)
        if dev_acc > best_acc:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = dev_acc

    end = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(end // 60, end % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, train_loader, dev_loader, criterion, optimizer_ft, num_epochs=10)
