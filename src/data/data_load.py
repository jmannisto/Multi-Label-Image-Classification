import numpy as np
import os
from glob import glob
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torch.utils.data import DataSet

data = pd.read_pickle("../../data/interim/DL_project_dataframe.pickle")
classes = ['clouds','male', 'bird', 'dog', 'river', 'portrait', 'baby', 'night', 'people', 'female',
           'sea', 'tree', 'car', 'flower']

def _get_data(img_path: str):
    #TODO: since everything is in the same folder path, we should use that and join the file names to simplify
    #TODO: see if we actually want to have the file path passed through here, or what other args we want
    if len(os.listdir(img_path)) == 0: #if the images folder is empty, then the actions have already been executed
        train_df = pd.read_pickle("../../data/interim/train_df.pickle")
        dev_df = pd.read_pickle("../../data/interim/valid_df.pickle")
        test_df= pd.read_pickle("../../data/interim/test_df.pickle")
        return train_df, dev_df, test_df
    else:
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
        dev_df = _make_dataset(shuffle, 0, int(len(files)*.1), '../../data/interim/valid/')
        test_df = _make_dataset(shuffle, int(len(files)*.1), int(len(files)*.2), '../../data/interim/test/')
        train_df = _make_dataset(shuffle, int(len(files)*.2), 20000, '../../data/interim/train/')

        #save these in interim folder
        train_df.to_pickle("../../data/interim/train_df.pickle")
        dev_df.to_pickle("../../data/interim/valid_df.pickle")
        test_df.to_pickle("../../data/interim/test_df.pickle")
    return train_df, dev_df, test_df

def _make_dataset(shuffle, start, end, dest_path):
    #TODO: Cite, heavily inspired by: https://thevatsalsaglani.medium.com/training-and-deploying-a-multi-label-image-classifier-using-pytorch-flask-reactjs-and-firebase-c39c96f9c427
    data_dict = {}
    data_file_names = []
    for i in shuffle[start:end]:
        file_name = files[i].split('/')[-1]
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
            f"{self.folder_dir}/{self.file_names[index]}")
        #cite: https://discuss.pytorch.org/t/convert-grayscale-images-to-rgb/113422/2
        convert = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        image = convert(image)
        label = self.labels[index]
        #image_pic = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        #TODO: convert label to tensor
        sample = {'image': image, 'label': np.array(label, dtype=np.float32)}
        if self.transform:
            image = self.transform(sample['image'])
            sample = {'image': image, 'label': label}

        return sample

def get_dataset(bs):
    #TODO: figure out what are actual parameters we want here
    #TODO: I want to set it up such that we can change the randomness and have different datasets pulled each time...
    #train_df = pd.read_pickle("../../data/interim/train_df.pickle")
    #dev_df = pd.read_pickle("../../data/interim/valid_df.pickle")
    #test_df= pd.read_pickle("../../data/interim/test_df.pickle")

    train_df, dev_df, test_df = _get_data('../../data/external/images/')

    print("Getting datasets")
    #TODO: transformations to do? NORMALIZE, resize? RandomRotation,
    train_dataset = MultiLabelData(train_df, '../../data/interim/train/')
    dev_dataset = MultiLabelData(dev_df, '../../data/interim/valid/')
    test_dataset = MultiLabelData(test_df, '../../data/interim/test/')

    print("Getting loaders")
    #TODO: possibly edit batch size based on argument
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size = bs, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    return train_loader, dev_loader, test_loader
