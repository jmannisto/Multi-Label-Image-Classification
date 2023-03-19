import pandas as pd
import numpy as np
import os
import glob

label_list = {} #list to store all files

directory = "../data/external/annotations" #"./dl2021-image-corpus-proj/annotations"
for n,filename in enumerate(glob.iglob(f'{directory}/*')):
    with open(filename) as f:
        label_list[filename.split('/')[-1]] = f.read().strip().split('\n')
        #TODO: figure out how to get the .txt out

new_keys = ['clouds','','male', 'bird', 'dog', 'river', 'portrait', 'baby', 'night', 'people', 'female',
            'sea', 'tree', 'car', 'flower']

#make a dataframe, rows are images, columns are features
data = pd.DataFrame(0,index=range(20000), columns= new_keys)

#set to correspond to image values
data = data.set_index(data.index+1)

#multihot encoding
for index, key in enumerate(label_list):
    for item in label_list[key]:
        data.at[int(item), new_keys[index]]=1

#add image names
filename = [os.path.basename(x) for x in glob.glob('../../data/external/images/*')]
data['image_name'] = filename
#save dataframe to pickle
data.to_pickle("../../data/interim/DL_project_dataframe.pickle")

#save dataframe to csv
data.to_csv("../../data/interim/DL_project_dataframe.csv")
