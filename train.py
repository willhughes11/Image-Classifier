import torch
from torch import nn, optim
from torchvision import models

from utility_functions import data_transforms,data_loader,model_data
from model_functions import save_checkpoint,validation,train_classifer,classifier_test

import argparse

parser = argparse.ArgumentParser(description="Image Classifier")
parser.add_argument('--arch', type = str, default = 'vgg', help = 'NN Model Architecture')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Learning Rate')
parser.add_argument('--hidden_units', type = int, default = 4096, help = 'Neurons in the Hidden Layer')
parser.add_argument('--epochs', type = int, default = 5, help = 'Epochs')
parser.add_argument('--gpu', type = str, default = 'cuda', help = 'GPU or CPU')
parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path to checkpoint')
arguments, unknown = parser.parse_known_args()

train_transforms,valid_transforms,test_transforms = data_transforms()
train_data,valid_data,test_data = data_loader(train_transforms,valid_transforms,test_transforms)
trainloader,validloader,testloader = model_data(train_data,valid_data,test_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if arguments.arch == 'vgg':
    input_size = 25088
    model = models.vgg16(pretrained=True)
else:
    input_size = 25088
    model = models.vgg16(pretrained=True)


for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(input_size,arguments.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(arguments.hidden_units,1000),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model.to(device)

print('----------Starting Validation---------')
validation(model,validloader,device,criterion)
print('----------Starting Training---------')
train_classifer(model,trainloader,arguments.epochs,device,optimizer,criterion,validloader)
print('----------Starting Testing---------')
classifier_test(model,device,testloader,criterion)
print('----------Saving Checkpoint---------')
save_checkpoint(model,train_data,arguments.arch,arguments.epochs,arguments.learning_rate,arguments.hidden_units,input_size)
print('----------Finished Model---------')