import torch
from torch import nn
from torchvision import models

def save_checkpoint(model, training_dataset, arch, epochs, lr, hidden_units, input_size):

    model.class_to_idx = training_dataset.class_to_idx
    checkpoint = {'input_size': (3, 224, 224),
                  'output_size': 102,
                  'hidden_layer_units': hidden_units,
                  'batch_size': 64,
                  'learning_rate': lr,
                  'model_name': arch,
                  'model_state_dict': model.state_dict(),
                  'epochs': epochs,
                  'class_to_idx': model.class_to_idx,
                  'clf_input': input_size}

    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['model_name'] == 'vgg':
        model = models.vgg16(pretrained=True)
     
    elif checkpoint['model_name'] == 'densenet':  
        model = models.densenet121(pretrained=True)
    else:
        print("Model Not Recognized")

    for param in model.parameters():
        param.required_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = nn.Sequential(nn.Linear(25088,4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(4096,1000),
                                 nn.LogSoftmax(dim=1))

    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def validation(model, validloader, device, criterion):
    loss = 0
    accuracy = 0
    batch_loss = 0
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model(inputs)
        batch_loss += criterion(logps, labels)
        loss += batch_loss.item()
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
    return loss, accuracy


def train_classifer(model,trainloader,arg_epochs,device,optimizer,criterion,validloader):
    epochs = arg_epochs
    steps = 0
    print_every = 25
    
    for epoch in range(epochs):
        
        running_loss = 0
        
        for inputs, labels in trainloader:
            #start = time.time()
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()

            #print(f'Training Seconds: {(time.time()- start):.5f} | Steps:{steps}')
            if steps % print_every == 0:
            
                model.eval()
                with torch.no_grad():
                
                  valid_loss,accuracy = validation(model,validloader,device,criterion)
                  
                print(f"Epoch: {epoch+1}/{epochs}.. ",
                      f"Training Loss: {(running_loss/len(trainloader)):.3f}.. ",
                      f"Validation Loss: {(valid_loss/len(validloader)):.3f}.. ",
                      f"Validation Accuracy: {(accuracy/len(validloader)):.3f}")
                model.train()

def classifier_test(model,device,testloader,criterion):
    model.eval()
    model.to(device)
    loss = 0
    batch_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs,labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)
            batch_loss += criterion(logps, labels)
                    
            loss += batch_loss.item()
                
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        print(f"Model Accuracy: {accuracy/len(testloader):.3f}")