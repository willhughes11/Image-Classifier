import torch
import argparse
import random
import os

from torch import nn, optim
from torchvision import models
from utility_functions import open_json
from model_functions import save_checkpoint,validation,train_classifer,classifier_test

data_dir = 'flower_data'
random_folder = random.randint(1,102)
random_image = random.choice(os.listdir(f'{data_dir}/test/{random_folder}'))
random_flower = f'{data_dir}/test/{random_folder}/{random_image}'

parser = argparse.ArgumentParser(description="Image Classifier")
parser.add_argument('--image_dir', type = str, default = random_flower, help = 'Path to image')
parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')
parser.add_argument('--topk', type = int, default = 5, help = 'Top N Classes and Probabilities')
parser.add_argument('--json', type = str, default = 'flower_to_name.json', help = 'class_to_name json file')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
arguments = parser.parse_args()