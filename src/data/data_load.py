import numpy as np
import pandas as pd
import os
from glob import glob
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

data = pd.read_pickle("../../data/interim/DL_project_dataframe.pickle")
classes = ['clouds','male', 'bird', 'dog', 'river', 'portrait', 'baby', 'night', 'people', 'female',
           'sea', 'tree', 'car', 'flower']

#TODO: consider having a specific type of shuffle such that each category is somewhat well represented across all?
#-- train-dev-test split --#
#split = .8/.1/.1
files = glob('../../data/external/images/*.jpg')
shuffle = np.random.RandomState(seed=42).permutation(len(files))
try:
    for i in ['train', 'valid', 'test']:
        os.mkdir(os.path.join('../../data/interim/', i))
except FileExistsError:
    pass

#TODO: Cite, heavily inspired by: https://thevatsalsaglani.medium.com/training-and-deploying-a-multi-label-image-classifier-using-pytorch-flask-reactjs-and-firebase-c39c96f9c427
#Valid set up
valid_dict = {}
valid_file_names = []
for i in shuffle[:2000]:
    file_name = os.path.basename(files[i]) #not exactly sure what this does
    labels = np.array(data[data['image_name']==file_name][classes]).tolist() #do I actually need the numpy arr thing?
    valid_dict[file_name] = labels
    valid_file_names.append(file_name)
    os.rename(files[i], os.path.join('../../data/interim/valid', file_name))
valid_df = pd.DataFrame.from_dict(valid_dict, orient='index', columns = ['labels'])

#test set up
test_dict = {}
test_file_names = []
for i in shuffle[2000:4000]:
    file_name = os.path.basename(files[i]) #get name after final /
    labels = np.array(data[data['image_name'] == file_name][classes]).tolist()
    test_dict[file_name] = labels
    test_file_names.append(file_name)
    os.rename(files[i], os.path.join('../../data/interim/test', file_name))
test_df = pd.DataFrame.from_dict(test_dict, orient='index', columns=['labels'])

#train set up
train_dict = {}
train_file_names = []
for i in shuffle[4000:]:
    file_name = os.path.basename(files[i])
    labels = np.array(data[data['image_name']==file_name][classes]).tolist()
    train_dict[file_name] = labels
    train_file_names.append(file_name)
    os.rename(files[i], os.path.join('../../data/interim/train', file_name))
train_df= pd.DataFrame.from_dict(train_dict, orient='index', columns = ['labels'])


# CITE: https://thevatsalsaglani.medium.com/training-and-deploying-a-multi-label-image-classifier-using-pytorch-flask-reactjs-and-firebase-c39c96f9c427

class MultiLabelData(Dataset):

    def __init__(self, dataframe, folder_dir, transform=None):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.transform = transform
        self.file_names = dataframe.index
        self.labels = dataframe['labels'].values.tolist()  # maybe add .tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        #TODO: see if performance changes if everything is converted to greyscale
        image = read_image(
            f"{self.folder_dir}/{self.file_names[index]}")
        #cite: https://discuss.pytorch.org/t/convert-grayscale-images-to-rgb/113422/2
        convert = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        image = convert(image)
        label = self.labels[index]
        #image_pic = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        sample = {'image': image, 'label': np.array(label, dtype=np.float32)}#, 'image_pic': image_pic}  # .astype(float)}
        if self.transform:
            image = self.transform(sample['image'])
            sample = {'image': image, 'label': label}  # .astype(float)}

        return sample

valid_data = MultiLabelData(valid_df, '../data/interim/valid/')
test_data = MultiLabelData(test_df, '../data/interim/test/')
train_data = MultiLabelData(train_df, '../data/interim/train/')

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)
valid_dataloader = DataLoader(valid_data, batch_size = 32, shuffle = False)

