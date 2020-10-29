import torch
from torch import nn
from torchvision import models

def save_checkpoint(model,train_data):
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
        'arch': 'vgg16',
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict() 
    }

    torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)

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

def train_classifer(model,trainloader,device,optimizer,criterion,validloader):
    epochs = 5
    steps = 0
    print_every = 25
    train_losses, valid_losses = [], []
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
                
                  valid_loss,accuracy = validation(model, validloader, criterion)
                  
                  train_losses.append(running_loss/len(trainloader))
                  valid_losses.append(valid_loss/len(validloader))

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
        print(f"Network Accuracy: {accuracy/len(testloader):.3f}")

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