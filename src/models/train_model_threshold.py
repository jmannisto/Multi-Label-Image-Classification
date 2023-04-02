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
import data_load

#---hyperparameters---
N_EPOCHS = 30
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
LR = 0.002
MOMENTUM = 0.9
NUM_CLASSES = 14
PREDICTION_THRESHOLD = 0.5 #hyperparameter for setting positive label threshold as opposed to rounding

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def initialize_model(feature_extract, use_pretrained=True):
    model_ft = models.vgg11(pretrained=use_pretrained) 
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    input_size = 224 
    return model_ft, input_size

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model_ft, input_size = initialize_model(feature_extract=False, use_pretrained=True)
model_ft = model_ft.to(device)
optimizer = optim.SGD(model_ft.parameters(), lr=LR, momentum=MOMENTUM)

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

#---set positive weights ofr loss function---
POS_WEIGHT = np.zeros(NUM_CLASSES)
for i in range(NUM_CLASSES):
    y = [sample[i] for sample in train_loader.dataset.labels]
    if y.count(1) == 0:
        POS_WEIGHT[i] = 1
    else:
        POS_WEIGHT[i] = y.count(0)/y.count(1)

POS_WEIGHT = torch.tensor(POS_WEIGHT).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)

def evaluate(model, loader, criterion, threshold=PREDICTION_THRESHOLD):
    evaluation_loss = 0.0
    evaluation_corrects = 0
    evaluation_total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            evaluation_loss += loss.item() * inputs.size(0)
            predicted = np.zeros(outputs.shape)
            for i, sample in enumerate(outputs):
                for j, value in enumerate(sample):
                    if value >= threshold:
                        predicted[i, j] = 1
            predicted = torch.tensor(predicted).to(device)
            evaluation_corrects += (predicted == labels).sum().item()
            evaluation_total += labels.size(0)
    return evaluation_loss/len(loader.dataset), evaluation_corrects/(evaluation_total*NUM_CLASSES)*100

def evaluation_report(model, loader, criterion, threshold=PREDICTION_THRESHOLD):
    all_predicted = torch.tensor(np.zeros((1, NUM_CLASSES))).to(device)
    all_labels = torch.tensor(np.zeros((1, NUM_CLASSES))).to(device)
    evaluation_loss = 0.0
    evaluation_corrects = 0
    evaluation_total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            evaluation_loss += loss.item() * inputs.size(0)
            predicted = np.zeros(outputs.shape)
            for i, sample in enumerate(outputs):
                for j, value in enumerate(sample):
                    if value >= threshold:
                        predicted[i, j] = 1
            predicted = torch.tensor(predicted).to(device)
            all_predicted = torch.cat((all_predicted, predicted), 0)
            all_labels = torch.cat((all_labels, labels), 0)
            evaluation_corrects += (predicted == labels).sum().item()
            evaluation_total += labels.size(0)
            report = metrics.classification_report(all_labels[1:].cpu(), all_predicted[1:].cpu(), zero_division=0)
    return evaluation_loss/len(loader.dataset), evaluation_corrects/(evaluation_total*NUM_CLASSES)*100, report


def train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=10):
    start = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        running_total = 0

        for batch in train_loader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predicted = np.zeros(outputs.shape)
            for i, sample in enumerate(outputs):
                for j, value in enumerate(sample):
                    if value >= PREDICTION_THRESHOLD:
                        predicted[i, j] = 1
            predicted = torch.tensor(predicted).to(device)
            running_corrects += (predicted == labels).sum().item()
            running_total += labels.size(0)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss/len(train_loader.dataset)
        epoch_acc = running_corrects/(running_total*NUM_CLASSES)*100

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion)
        print('Development Loss: {:.4f} Development Acc: {:.4f}'.format(dev_loss, dev_acc))

    end = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(end // 60, end % 60))
    return model

model_ft = train_model(model_ft, train_loader, dev_loader, criterion, optimizer_ft, num_epochs=N_EPOCHS)

for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    test_loss, test_acc, test_report = evaluation_report(model_ft, test_loader, criterion, threshold=t)
    print("Test results with prediction threshold " + str(t))
    print('Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))
    print(test_report)
torch.save(model_ft, "modelfile.pt")