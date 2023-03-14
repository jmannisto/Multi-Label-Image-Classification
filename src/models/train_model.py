import torch
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data_load

#TODO: experiment with optimizers
#TODO: pretrained models?
#TODO: actually set up so it interacts with data_load
#

# --- hyperparameters ---
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32
LR = 0.001

# --- fixed constants ---
NUM_CLASSES = 14
DATA_DIR = '../../data/interim/'
TRAIN_TEST_DEV_RATIO = [0.8, 0.1, 0.1] #TODO: pass this through to data load

# --- Dataset initialization ---
#TODO: reset data initialization

print('Preparing dataset')
train_loader, dev_loader, test_loader = data_load.get_dataset(BATCH_SIZE_TRAIN)

# --- eval functions ---
def accuracy():
    #how to evaluate?
    pass

def evaluation():
    #dev testing
    #inference time, on CPU and GPU
    #compare model sizes
    #
    pass

# --- model ---
#TODO: rewrite CNN model
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

        self.layer_fc = nn.Sequential(nn.Flatten(), nn.Linear(576, 500), nn.ReLU(), nn.Linear(500, num_classes))

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer_fc(output)
        return output


# --- set up ---
# best performing set up is loaded, but additional options are included in comments or noted in comments above
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CNN() #.to(device)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
loss_function = nn.BCEWithLogitsLoss()

# --- training ---
dev_err = []
for epoch in range(N_EPOCHS):
    train_loss = 0
    train_correct = 0
    total = 0
    epoch_loss =0
    for batch in train_dataloader:
        image, label = batch['image'], batch['label'] #include t.to(device)
        # calculate loss and calculate total loss and total correct
        loss = loss_function(model(image), label)
        epoch_loss += loss.item()
        train_correct = (torch.argmax(model(image), dim=1) == target).sum().item()

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #TODO: fix the metrics printed out here, loss is not correct and accuracy is not set up
        print('Training: Epoch %d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
              (epoch, epoch_loss / (len(train_dataloader)),
               100. * train_correct / total, train_correct, total))
    train_loss = (epoch_loss / len(train_dataloader))

    dev_loss = 0
    dev_correct = 0
    size = len(dev_loader.dataset)
    num_batches = len(dev_loader)
    with torch.no_grad():
        for data, target in dev_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            dev_loss += loss_function(pred, target).item()
            dev_correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    dev_loss /= num_batches
    dev_correct /= size
    print("Dev Loss:", dev_loss)
    dev_err.append(dev_loss)

    # Please implement early stopping here.
    # You can try different versions, simplest way is to calculate the dev error and
    # compare this with the previous dev error, stopping if the error has grown.
    # add early stopping
    #TODO: make dev error based on more than just 1 decrease, and maybe also based on % instead of >
    if len(dev_err) > 2:
        if dev_err[-1] > dev_err[-2]:
            break

# --- test ---
test_loss = 0
test_correct = 0
total = 0
# added
avg_acc = []
model.eval()
with torch.no_grad():
    for batch_num, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        loss = loss_function(model(data), target)
        test_loss += loss.item()
        total = target.size(0)
        test_correct = (torch.argmax(model(data), dim=1) == target).sum().item()
        avg_acc.append(100. * test_correct / total)

        print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' %
              (batch_num, len(test_loader), test_loss / (batch_num + 1),
               100. * test_correct / total, test_correct, total))

print("Avg Accuracy: ", (sum(avg_acc) / len(avg_acc)))
