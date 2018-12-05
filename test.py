from __future__ import print_function
from __future__ import division

import torch
import torchvision.transforms as transforms

import numpy as np
from scipy.misc import imread, imresize

from models.AlexNet import *
from models.ResNet import *
import os
import dataset


def load_model(model_name):
    """load the pre-trained model"""
    if model_name == 'ResNet':
        model = resnet_18()
        model_path = './models/resnet.pt'
    elif model_name == 'AlexNet':
        model = alexnet()
        model_path = './models/alexnet.pt'
    elif model_name == 'OurBest':
        model = resnet_18()
        model_path = './models/best/model.10'
    else:
        raise NotImplementedError(model_name + ' is not implemented here')

    checkpoint = torch.load(model_path, map_location='cpu')
    # import pdb; pdb.set_trace()
    model.load_state_dict(checkpoint)
    return model


def construct_transformer():
    """construct transformer for images"""
    mean = [0.45486851, 0.43632515, 0.40461355]
    std = [0.26440552, 0.26142306, 0.27963778]
    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transformer


def load_categories():
    """load the classification id-name dictionary"""
    categories = list([])
    file = open('./categories.txt', 'r')
    for line in file:
        words = line.split(' ')
        categories.append(words[0])
    return categories

def load_validation_data():
    """load the image paths and groundtruth categories in val.txt into a map"""
    validation_data = {}
    file = open('val.txt', 'r')
    for line in file:
        words = line.split(' ')
        validation_data[words[0]] = int(words[1])
    return validation_data

def main():
    categories = load_categories()

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 100

    # load model and set to evaluation mode
    network = 'OurBest'
    # network = 'AlexNet'
    model = load_model(network)

    model.to(device)
    model.eval()
    val_loader, test_loader = dataset.get_val_test_loaders(batch_size)

    ###################################################
    # Making result file
    ###################################################


    f= open("2pm/result.txt","w")
    f_ = open("2pm/result_more.txt","w")
    count = 1
    for batch_num, (inputs, labels) in enumerate(test_loader, 1):
        inputs = inputs.to(device)
        prediction = model(inputs)
        prediction = prediction.to('cpu')
        _, ind = torch.topk(prediction,5)
        
        for i in xrange(ind.shape[0]):
            line = 'test/' + '%08d' % count + '.jpg ' + str(ind[i][0].item()) + ' ' + str(ind[i][1].item()) + ' ' + str(ind[i][2].item()) + ' ' + str(ind[i][3].item()) + ' ' + str(ind[i][4].item()) +'\n'
            line_details = 'test/' + '%08d' % count  + '.jpg ' + str(categories[ind[i][0].item()]) + ' ' + str(categories[ind[i][1].item()]) + ' ' + str(categories[ind[i][2].item()]) + ' ' + str(categories[ind[i][3].item()]) + ' ' + str(categories[ind[i][4].item()]) +'\n'
            f.write(line)
            f_.write(line_details)
            count += 1
    f.close()
    f_.close()




 

main()
