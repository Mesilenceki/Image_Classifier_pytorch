import argparse

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# input the arguments that need to train the parameters
parser = argparse.ArgumentParser(description = 'Input the parameters to Train a Neural Network')
parser.add_argument('-data_directory', dest ='data_directory', help = 'Enter path to training data.')
parser.add_argument('-arch', action='store',dest = 'arch_name', 
                    help= 'Enter pretrained model to use;choices have:vgg16,resnet16,dense')
parser.add_argument('-save_dir', action = 'store',dest = 'save_directory', 
                    help = 'Enter location to save checkpoint in.')
parser.add_argument('-learning_rate', action = 'store',dest = 'lr', type= float,
                    help = 'Enter learning rate for training the model.')
parser.add_argument('-hidden_units_1', action = 'store', dest = 'units_1', type= int,
                    help = 'Enter number of hidden units in layer1.')
parser.add_argument('-hidden_units_2', action = 'store', dest = 'units_2', type= int, 
                    help = 'Enter number of hidden units in layer2.')
parser.add_argument('-epochs', action = 'store', dest = 'num_epochs', type = int, 
                    help = 'Enter number of epochs to use during training,.')
parser.add_argument('-gpu', action = "store", dest = 'device',
                    help = 'choice are cuda or cpu')
parameters = parser.parse_args()

data_dir = parameters.data_directory
save_dir = parameters.save_directory
arch_name = parameters.arch_name
lr = parameters.lr
hidden_layer_1 = parameters.units_1
hidden_layer_2 = parameters.units_2
epochs = parameters.num_epochs
gpu_mode = parameters.device

#get the data that used to train the model
def get_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = {
        'training': transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
        'testing': transforms.Compose([transforms.transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]),
        'validation': transforms.Compose([transforms.transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'training': datasets.ImageFolder(train_dir, transform= data_transforms['training']),
        'testing':datasets.ImageFolder(test_dir, transform= data_transforms['testing']),
        'validation':datasets.ImageFolder(valid_dir, transform= data_transforms['validation'])
    
    }

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'training': torch.utils.data.DataLoader(image_datasets['training'], batch_size= 64, shuffle= True),
        'testing': torch.utils.data.DataLoader(image_datasets['testing'], batch_size= 64, shuffle= True),
        'validation':torch.utils.data.DataLoader(image_datasets['validation'], batch_size= 64, shuffle= True)
    
    }
    return dataloaders['training'], dataloaders['validation'], dataloaders['testing']
# Build and attach new classifier
names = {"vgg16":25088,
        "densenet161": 2208,
        "SqueezeNet":512}
def classifier(name , hidden_layer_1, hidden_layer_2, lr, device):
    # Use cuda or cpu according to user's choice 
    device = torch.device(device)
    if name == 'vgg16':
        model = models.vgg16(pretrained= True)
    elif name == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif name == 'SqueezeNet':
        name = models.SqueezNet1_0(pretrained=True)
    else:
        print("please use vgg16,densenet16 or Squeezenet")
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Continue to define my new classification network   
    model.classifier = nn.Sequential(nn.Linear(names[name], hidden_layer_1),
                                     nn.ReLU(),
                                     nn.Dropout(p = 0.5),
                                     nn.Linear(hidden_layer_1, hidden_layer_2),
                                     nn.ReLU(),
                                     nn.Dropout(p = 0.5),
                                     nn.Linear(hidden_layer_2, 102),
                                     nn.LogSoftmax(dim = 1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    model.to(device)
    return model, criterion, optimizer
# Train model
def train(model, epochs, trainloaders, valloaders, optimizer, criterion, device):    
    # Use cuda or cpu according to user's choice
    device = torch.device(device)
    steps = 0
    print_every = 80
    train_losses, val_losses = [],[]
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for inputs,labels in iter(trainloaders):
            steps +=1
            inputs, labels = inputs.to(device), labels.to(device) #save data to device
            optimizer.zero_grad()
            logit = model.forward(inputs)
            loss = criterion(logit, labels) #calculate the loss
            loss.backward()
            optimizer.step()  
            running_loss += loss.item()
            
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in iter(valloaders):
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = model.forward(inputs)
                        batch_loss = criterion(output, labels)
                        val_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                        train_losses.append(running_loss/len(trainloaders))
                        val_losses.append(val_loss/len(valloaders))    
                        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                              "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                              "Val Loss: {:.3f}.. ".format(val_loss/len(valloaders)),
                              "Val Accuracy: {:.3f}".format(accuracy/len(valloaders)))
    return model

# Test model
def test(model, testloaders, criterion, device):
    # Use GPU if it's available
    device = torch.device(device)
    accuracy = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloaders:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            batch_loss = criterion(output, labels)
            test_loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()    
            print("Test Loss: {:.3f}.. ".format(test_loss/len(testloaders)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloaders)))

# Save model
def save_checkpoint(model,save_dir):
    checkpoint = {'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'class_to_idx': t.class_to_idx,
              'opt_state': optimizer.state_dict}
    torch.save(checkpoint, save_dir)


def main():
    trainloader, valloader, testloader = get_data(data_dir)
    model, criterion, optimizer = classifier(arch_name,hidden_layer_1, hidden_layer_2, lr, gpu)
    train(model, epochs,trainloader, valloader,  optimizer, criterion, gpu)
    test(model, testloader, criterion, gpu)
    save_checkpoint(model, save_dir)

if __name__ == '__main__':
    main()
