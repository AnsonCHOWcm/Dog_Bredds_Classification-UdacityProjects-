#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:45:05 2021

@author: ccm
"""
import numpy as np
import cv2
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt
                      
## Face-Detector

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

### Dog-Dectector

# define VGG16 model
VGG16 = models.vgg16(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    VGG16 = VGG16.cuda()
    
def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    img = Image.open(img_path)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean = [0.485 , 0.456 , 0.406],
        std = [0.229 , 0.224 , 0.225])
    ])
    
    input_img = transform(img)
    
    batched_img = torch.unsqueeze(input_img , 0)
    
    if use_cuda:
        batched_img = batched_img.cuda()
    
    output = VGG16(batched_img)
    
    _ , top_class = torch.max(output , 1 )
    
    
    return top_class # predicted class index

def dog_detector(img_path):
    ##  Complete the function
    
    return 1 if (VGG16_predict(img_path) >= 151 and VGG16_predict(img_path) <= 268 ) else 0

## Def the function for training modela nd testing Model

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            optimizer.zero_grad()
            
            log_ps = model(data)
            loss = criterion(log_ps , target)
            loss.backward()
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            log_ps = model(data)
            loss = criterion(log_ps , target)
            
            valid_loss  = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## save the model if validation loss has decreased
        if valid_loss < valid_loss_min :
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
            
    # return trained model
    return model

def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))




## Loading the Dataset for Training , Vailding and Testing

transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
        mean = [0.485 , 0.456 , 0.406],
        std = [0.229 , 0.224 , 0.225])
    ])

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean = [0.485 , 0.456 , 0.406],
        std = [0.229 , 0.224 , 0.225])
    ])



training_set = datasets.ImageFolder("./data/dog_images/train", transform=transform_train)
testing_set = datasets.ImageFolder("./data/dog_images/test", transform=transform)
validing_set = datasets.ImageFolder("./data/dog_images/valid", transform=transform)

trainloader = torch.utils.data.DataLoader(training_set, batch_size=64 , shuffle = True)
testloader = torch.utils.data.DataLoader(testing_set , batch_size=64, shuffle = True)
validloader = torch.utils.data.DataLoader(validing_set, batch_size=64, shuffle = True)

loaders = {'train' : trainloader ,
           'test' : testloader ,
           'valid' : validloader}

data = {"train" : training_set , "test" : testing_set , "valid" :validing_set }



## Build the Dog Breed Classifier from scratch

# define the CNN architecture
class Net(nn.Module):
    ### choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        
        self.fc1 = nn.Linear(784 * 128 , 512)
        self.fc2 = nn.Linear(512,133)
        
        self.pooling = nn.MaxPool2d(2,2)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        ## Define forward behavior
        
        x = self.pooling(F.relu(self.conv1(x)))
        x = self.pooling(F.relu(self.conv2(x)))
        x = self.pooling(F.relu(self.conv3(x)))
        x = x.view(x.shape[0] , -1)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x) , dim =1)
        
                        
        
        return x


# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
    
###  select loss function
criterion_scratch = nn.NLLLoss()

###  select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters() , lr = 0.01)

## Training the scratch Model

# train the model
model_scratch = train(40, loaders, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda, 'model_scratch.pt')

# load the model that got the best validation accuracy
model_scratch.load_state_dict(torch.load('model_scratch.pt'))

## Testing the model

test(loaders, model_scratch, criterion_scratch, use_cuda)

## Build the Cod-Breed Classifier from transfer learning

## Specify model architecture 

model_transfer = models.resnet50(pretrained=True)

for param in model_transfer.parameters():
    param.requires_grad = False
    
model_transfer.fc = nn.Sequential(nn.Linear(2048, 512),
                                  nn.ReLU(),
                                  nn.Dropout(0.5),
                                  nn.Linear(512 , 133)
                                          )



if use_cuda:
    model_transfer = model_transfer.cuda()
    
## Specify the loss function and Gradient Optimization Method

criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.RMSprop(model_transfer.fc.parameters(), lr = 0.001)

# train the model
model_transfer =  train(10, loaders, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')

# load the model that got the best validation accuracy (uncomment the line below)
#model_transfer.load_state_dict(torch.load('model_transfer.pt'))

model_transfer.load_state_dict(torch.load('model_transfer.pt'))

# Test the Model

test(loaders, model_transfer, criterion_transfer, use_cuda)

### Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in data['train'].classes]

def predict_breed_transfer(img_path):
    
    img = Image.open(img_path)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean = [0.485 , 0.456 , 0.406],
        std = [0.229 , 0.224 , 0.225])
    ])
    
    input_img = transform(img)
    
    batched_img = torch.unsqueeze(input_img , 0)
    
    if use_cuda:
        batched_img = batched_img.cuda()
    
    output = model_transfer(batched_img)
    
    _ , top_class = torch.max(output , 1 )
    
    
    return class_names[top_class]

### Write algorithm.

def run_app(img_path):
    ## handle cases for a human face, dog, and neither
    
    if dog_detector(img_path):
        
        img = Image.open(img_path)
        
        output = predict_breed_transfer(img_path)
        
        print('Hi ! Dogge')
        plt.imshow(img)
        plt.show()
        print("You look like a ..." , output)
        
    elif face_detector(img_path):
        
        img = Image.open(img_path)
        
        output = predict_breed_transfer(img_path)
        
        print('Hi ! Human')
        plt.imshow(img)
        plt.show()
        print("You look like a ..." , output)
        
    else :
        
        img = Image.open(img_path)
        output = "Neither Human nor Dog is detected in the photo"
        plt.imshow(img)
        plt.show()
        print("You look like a ..." , output)

#Test with user-input        

for file in ["./Elon.jpg" , "./Maria.jpeg" , "./Bulldog.jpg" , "./Poodle.jpg" , "./DeepLearning.jpeg" , "./udacity.jpg"]:
    run_app(file)