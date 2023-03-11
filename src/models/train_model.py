import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#TODO: actually set up so it interacts with data_load

# --- hyperparameters ---
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
LR = 0.001

# --- fixed constants ---
NUM_CLASSES = 14
DATA_DIR = '../../data/interim/'

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
valid_dataloader = DataLoader(valid_dataset, batch_size = 32, shuffle = False)

#basic CNN model, default nothing special
class CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNN, self).__init__()
        # input shape: (100,3, 28, 28) i.e. (BATCH_SIZE, CHANNELS, WIDTH, HEIGHT)
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                    nn.Dropout(p=0.25))  # dropout wasn't included in all tests

        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3),
                                    nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                    nn.Dropout(p=0.25))  # dropout wasn't included in all test

        self.layer_fc = nn.Sequential(nn.Flatten(), nn.Linear(15376, 500), nn.ReLU(), nn.Linear(500, num_classes))

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer_fc(output)
        return output

model = CNN()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
loss_function = nn.BCEWithLogitsLoss()

dev_err = []
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0
    epoch_loss =0
    for batch in train_dataloader:
        image, label = batch['image'], batch['label']
        # calculate loss and calculate total loss and total correct
        loss = loss_function(model(image), label)
        epoch_loss += loss.item()
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = (epoch_loss / len(train_dataloader))
    print(train_loss)
    
   
