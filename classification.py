from __future__ import print_function
import os
import os.path
import sys
import random
import time

import numpy as np
import PIL
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torch.autograd
import torch.autograd.variable
import torchvision
import torchvision.transforms

#check cuda
seed = 1
if not torch.cuda.is_available():
    print("no cuda")
    quit()
torch.cuda.manual_seed(seed)

#load data
imsize=32
testdata = []
for i in range(4):
    l = os.listdir("data/sub_cifar10/"+str(i))
    l.sort()
    for f in l:
        #x: image  y: label
        testdata.append((np.asarray(PIL.Image.open("data/sub_cifar10/"+str(i)+"/"+f).convert("RGB").copy()),i))

#this is a non sense from a learning point of view, but here, we just learn on the testing data for simplicity 
traindata = testdata.copy()

#here we define a network for this inpainting problem
#note that writing the network is not a big deal
#the problem is to find the one on which optimization will behave correctly
#here a very simple network just to illustrate common operations
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, bias=True, kernel_size=11,padding=5)
        self.conv2 = nn.Conv2d(16, 16, bias=True, kernel_size=5,padding=2)
        self.fc1 = nn.Linear(1024, 64, bias=True)
        self.fc2 = nn.Linear(64, 4, bias=True)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net()
model.cuda()
model.train()

#define solver parameter, here we will have static parameter
lr = 0.001
momentum = 0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
losslayer = nn.CrossEntropyLoss()
os.makedirs("tmp")
lossduringtraining = open("tmp/lossduringtraining.txt","w")

############ def train test

batchsize = 16

def trainbatch(inputnumpytensor,targetnumpytensor):
    #basic foward backward routine
    #data are converted to allow backward
    #then we do the forward
    #then we compute the loss
    #then we compute the gradient
    #finaly we update the weight
    
    data = torch.autograd.Variable(torch.Tensor(inputnumpytensor))
    target = torch.autograd.Variable(torch.from_numpy(targetnumpytensor).long())
    data, target = data.cuda(), target.cuda()
    
    optimizer.zero_grad()
    output = model(data)
    
    loss = losslayer(output, target)
    lossduringtraining.write(str(loss.cpu().data.numpy()[0])+"\n")
    lossduringtraining.flush()
    
    loss.cuda()
    loss.backward()
    
    optimizer.step()

def train():
    random.shuffle(traindata)
    for i in range(0,len(traindata)-batchsize,batchsize):
        x = np.zeros((batchsize,3,imsize,imsize), dtype=float)
        y = np.zeros(batchsize, dtype=int)
        for j in range(batchsize):
            (xx,yy) = traindata[i+j]
            for r in range(imsize):
                for c in range(imsize):
                    for ch in range(3):
                        x[j][ch][r][c] = xx[r][c][ch]/16
            y[j] = yy
        trainbatch(x,y)

def test():
    for xx,y in testdata:
        x = np.zeros((1,3,imsize,imsize), dtype=float)
        for r in range(imsize):
            for c in range(imsize):
                for ch in range(3):
                    x[0][ch][r][c] = xx[r][c][ch]/16
                    
        data = torch.autograd.Variable(torch.Tensor(x)).cuda()
        
        output = model(data)
        prob = output.cpu().data.numpy()
        if prob[0][0]>=prob[0][1] and prob[0][0]>=prob[0][2] and prob[0][0]>=prob[0][3]:
            print(str(y)+" predicted as 0");
        if prob[0][1]>=prob[0][0] and prob[0][1]>=prob[0][2] and prob[0][1]>=prob[0][3]:
            print(str(y)+" predicted as 1");
        if prob[0][2]>=prob[0][0] and prob[0][2]>=prob[0][1] and prob[0][2]>=prob[0][3]:
            print(str(y)+" predicted as 2");
        if prob[0][3]>=prob[0][0] and prob[0][3]>=prob[0][1] and prob[0][3]>=prob[0][2]:
            print(str(y)+" predicted as 3");

print("start training testing")
for i in range(30):
    print(str(i))
    train()
test()
