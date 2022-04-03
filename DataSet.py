# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:02:33 2021

@author: AlexandreN
"""

import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import numpy as np


# all of the collected information on the images.
class QuestionDataSet(torch.utils.data.Dataset):
    """ init stage
     root : root of the data folder
     transform : what transformation will be applied to the data
     setting : 'training', 'validation' or 'test'
     training_proportion : proportion of data used for training among training data available
     only_hanging : filter the dataset to only images of person hanging on a pull-up bar
    """
    def __init__(self, root, transform, setting='training', training_proportion=1, only_hanging=False):

        torch.utils.data.Dataset.__init__(self)

        self.root = root
        if self.root[-1] != '/':
            self.root += '/'

        df = pd.read_csv(self.root + "all_labels.csv", low_memory=False)

        df = df.loc[df['phase'] == setting]
        if only_hanging:
            df = df.loc[df['traction'] == 1]

        df = df.reset_index(drop=True)

        # get the requested proportion of the Dataset (if we wish to train with a smaller dataset for example)
        # remark: should only apply to the training set
        if setting == "training":
            df = df.sample(frac=training_proportion).reset_index(drop=True)

        # setup the list of images and targets
        self.image_list = df.iloc[:, 0:2]
        self.target_df = df.iloc[:, 3:8]

        self.transform = transform

    def __len__(self):
        return self.target_df.shape[0]

    def __getitem__(self, index):
        image_path = self.root + 'images/' + self.image_list.iloc[index, 1] +\
                     '/' + self.image_list.iloc[index, 0] + '.jpg'

        image = Image.open(image_path)
        width, height = image.size

        labels = self.target_df.iloc[index, :]

        keypoints = []
        for i in range(1, 3):
            keypoints.append((labels[i * 2 - 1] * width,
                              labels[i * 2] * height))

        transformed = self.transform(image=np.array(image), keypoints=keypoints)
        if labels[0] == 0:
            target_list = [0, 0, 0, 0, 0]
        else:
            target_list = [labels[0]]
            for keypoint in transformed['keypoints']:
                for coord in keypoint:
                    target_list.append(coord)

        target = torch.Tensor(target_list)
        return transformed['image'], target

    # function to get the disparity between the number of positive and negative cases
    def get_pos_weight(self) -> torch.Tensor:
        total = self.target_df.shape[0]
        total_positive = int(self.target_df['traction'].sum())
        total_negative = total - total_positive
        pos_weight = torch.Tensor([total_negative / total_positive])
        return pos_weight
