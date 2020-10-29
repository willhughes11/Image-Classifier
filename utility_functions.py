import torch
from torchvision import transforms, datasets

import random
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from PIL import Image

data_dir = 'flower_data'
train_dir = f'{data_dir}/train'
valid_dir = f'{data_dir}/valid'
test_dir = f'{data_dir}/test'

def data_transforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30), 
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])
                                                        
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])

    return train_transforms,valid_transforms,test_transforms

def data_loader(train_transforms,valid_transforms,test_transforms):
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    return train_data,valid_data,test_data

def model_data(train_data,valid_data,test_data):
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return trainloader,validloader,testloader

def open_json(json_file):
    with open(json_file, 'r') as f:
        flower_to_name = json.load(f)
    
    return flower_to_name

def process_image(image):
    pil_image = Image.open(image)

    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((4096,256))
    else:
        pil_image.thumbnail((256,4096))
      
    left_margin = (pil_image.width - 224)/2
    bottom_margin = (pil_image.height - 224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224

    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))

    np_image = np.array(pil_image)/255
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    np_image = np_image.transpose((2,0,1))

    return np_image

def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def prediction_test(flower_to_name,classes,probs,random_folder):
    
    flower_title = flower_to_name[str(random_folder)].capitalize()
    flower_names = [flower_to_name[i] for i in classes]

    print('------- Model Prediction Test -------')
    print(f'Correct Flower Name: {flower_title}')
    print('-------- Probability --------')
    for i in range(len(probs)):
        print(f'{i+1} - Flower: {flower_names[i]}... Probability: {(probs[i]*100):.2f}%')

def predict(image_path, model, topk=5):
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    logps = model.forward(image)
    ps = torch.exp(logps)
    top_ps,top_indices = ps.topk(topk)

    top_ps = top_ps.detach().type(torch.FloatTensor).numpy().tolist()[0]
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0]

    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in top_indices]

    return top_ps, top_classes