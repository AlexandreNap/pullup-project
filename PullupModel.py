# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 15:07:27 2021

@author: AlexandreN
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torchvision


class PullupHead(nn.Module):

    def __init__(self):
        super(PullupHead, self).__init__()

        # head_class is the pull-up classification head
        self.head_class = nn.Sequential(nn.Linear(2048, 128),
                                        nn.ReLU(),                          
                                        nn.Dropout(p=0.3),
                                        nn.Linear(128, 1))

        # head_locs is the pull-up localisation head
        self.head_locs = nn.Sequential(nn.Linear(2048, 1024),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.3),
                                       nn.Linear(1024, 4),
                                       nn.Sigmoid()
                                       )

    def forward(self, features):
        features = features.view(features.size()[0], -1)
        
        y_bbox  = self.head_locs(features)
        y_class = self.head_class(features)

        res = (y_bbox, y_class)
        return res


def create_model():
    # setup the architecture of the model
    feature_extractor = torchvision.models.resnet50(pretrained=True)
    model_body = nn.Sequential(*list(feature_extractor.children())[:-1])
    # for param in model_body.parameters():
    #     param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    
    model_head = PullupHead()
    model = nn.Sequential(model_body, model_head)
    return model


def load_weights(model, path='output/model/model.pt', device_='cpu'):
    checkpoint = torch.load(path, map_location=torch.device(device_))
    model.load_state_dict(checkpoint)
    return model

