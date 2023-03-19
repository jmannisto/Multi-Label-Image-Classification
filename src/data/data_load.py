import numpy as np
import pandas as pd
import os
from glob import glob
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms 

data = pd.read_pickle("../../data/interim/DL_project_dataframe.pickle")
classes = ['clouds','male', 'bird', 'dog', 'river', 'portrait', 'baby', 'night', 'people', 'female',
           'sea', 'tree', 'car', 'flower']

def _get_data(img_path: str):
"""
loads the train, dev, and test data frames if already made
if not already made, calls _make_dataset to create them 
randomly shuffles the data, and splits in a .8-.1-.1 split, passes the corresponding start and end points for each set to the method
pickles created dataframes in data/interim
"""
    #TODO: since everything is in the same folder path, we should use that and join the file names to simplify
    #TODO: see if we actually want to have the file path passed through here, or what other args we want
    #file names were updated as they were causing problems when running in VSCode
    if len(os.listdir(img_path)) == 0: #if the images folder is empty, then the actions have already been executed
        file_path = os.path.join(os.path.dirname(__file__), '../../data/interim')
        train_df = pd.read_pickle(os.path.join(file_path, 'train_df.pickle'))
        dev_df = pd.read_pickle(os.path.join(file_path, 'valid_df.pickle'))
        test_df = pd.read_pickle(os.path.join(file_path, 'test_df.pickle'))
        return train_df, dev_df, test_df
    else:
        #TODO: edit the file paths for this option
        split = [0.8, 0.1, 0.1] #80-10-10 split
        # TODO: consider having a specific type of shuffle such that each category is somewhat well represented across all?
        files = glob('../../data/external/images/*.jpg')
        shuffle = np.random.RandomState(seed=42).permutation(len(files))
        # try to make train, valid, and test folders
        try:
            for i in ['train', 'valid', 'test']:
                os.mkdir(os.path.join('../../data/interim/', i))
        except:
            pass
        #TODO: connect with split ratio and lengths
        #make the dataframes for the 3 datasets
        dev_df = _make_dataset(shuffle, 0, int(len(files)*.1), '../../data/interim/valid/', files)
        test_df = _make_dataset(shuffle, int(len(files)*.1), int(len(files)*.2), '../../data/interim/test/', files)
        train_df = _make_dataset(shuffle, int(len(files)*.2), 20000, '../../data/interim/train/', files)

        #save these in interim folder
        train_df.to_pickle("../../data/interim/train_df.pickle")
        dev_df.to_pickle("../../data/interim/valid_df.pickle")
        test_df.to_pickle("../../data/interim/test_df.pickle")
    return train_df, dev_df, test_df

def _make_dataset(shuffle, start, end, dest_path, files):
"""
creates corresponding data frames with file name in one column and array of multi-hot encoded vectors in other column
image files are moved to their corresponding test, valid, or train folder in data/interim and the dataframes are pickled there too
"""
    #TODO: Cite, heavily inspired by: https://thevatsalsaglani.medium.com/training-and-deploying-a-multi-label-image-classifier-using-pytorch-flask-reactjs-and-firebase-c39c96f9c427
    data_dict = {}
    data_file_names = []
    for i in shuffle[start:end]:
        file_name = os.path.basename(files[i])
        labels = np.array(data[data['image_name'] == file_name][classes]).tolist() #actually need np.arr?
        data_dict[file_name] = labels
        data_file_names.append(file_name)
        os.rename(files[i], os.path.join(dest_path, file_name)) #('../../data/interim/valid', file_name))
    return pd.DataFrame.from_dict(data_dict, orient='index', columns=['labels'])


#TODO: CITE: https://thevatsalsaglani.medium.com/training-and-deploying-a-multi-label-image-classifier-using-pytorch-flask-reactjs-and-firebase-c39c96f9c427
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
            f"{self.folder_dir}/{self.file_names[index]}").float()
        #cite: https://discuss.pytorch.org/t/convert-grayscale-images-to-rgb/113422/2
        convert = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        image = convert(image)
        label = self.labels[index]
        #image_pic = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        #TODO: convert label to tensor
        sample = {'image': image, 'label': np.array(label, dtype=np.float32)}
        if self.transform:
            image = self.transform(sample['image'])
            sample = {'image': image, 'label': np.array(label, dtype=np.float32)}

        return sample

def get_dataset(bs, transforms = None):
    #TODO: figure out what are actual parameters we want here
    #TODO: I want to set it up such that we can change the randomness and have different datasets pulled each time...
    #added:
    full_path = os.path.dirname(__file__)
    train_df, dev_df, test_df = _get_data(os.path.join(os.path.dirname(__file__),'../../data/external/images/'))

    print("Getting datasets")
    train_dataset = MultiLabelData(train_df, (os.path.join(full_path,'../../data/interim/train/')), transforms)
    dev_dataset = MultiLabelData(dev_df, os.path.join(full_path, '../../data/interim/valid/'), transforms)
    test_dataset = MultiLabelData(test_df, os.path.join(full_path, '../../data/interim/test/'), transforms)

    print("Getting loaders")
    #TODO: possibly edit batch size based on argument
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size = bs, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    return train_loader, dev_loader, test_loader
