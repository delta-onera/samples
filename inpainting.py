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
imsize=128
testdata = []
for i in range(5):
    #x: image where some pixel have been set to black
    #y: real image
    testdata.append((PIL.Image.open("data/"+str(i)+"x.png").copy(),PIL.Image.open("data/"+str(i)+"y.png").copy()))
    
#this is a non sense from a learning point of view, but here, we just learn on the testing data for simplicity 
traindata = testdata.copy()

#create a tmp folder to store prediction
os.makedirs("tmp")
for i in range(len(testdata)):
    x,y = testdata[i]
    x.save("tmp/"+str(i)+"x.png")
    y.save("tmp/"+str(i)+"y.png")

#here we define a network for this inpainting problem
#note that writing the network is not a big deal
#the problem is to find the one on which optimization will behave correctly
#here a very simple network just to illustrate common operations
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(6, 32, kernel_size=3,padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3,padding=1, bias=True)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3,padding=1, bias=True)
        self.conv1_4 = nn.Conv2d(192, 64, kernel_size=1, bias=True)
        self.conv1_5 = nn.Conv2d(96, 1, kernel_size=1, bias=True)
        
        self.conv2_1 = nn.Conv2d(6, 32, kernel_size=3,padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3,padding=1, bias=True)
        self.conv2_3 = nn.Conv2d(64, 128, kernel_size=3,padding=1, bias=True)
        self.conv2_4 = nn.Conv2d(192, 64, kernel_size=1, bias=True)
        self.conv2_5 = nn.Conv2d(96, 1, kernel_size=1, bias=True)

    def forward(self, x):
        avg = F.avg_pool2d(x,kernel_size=7, stride=1, padding=3)
        
        x1_1 = torch.cat([x,avg],1)
        x1_1 = F.relu(self.conv1_1(x1_1))
        x1_2 = F.avg_pool2d(x1_1,kernel_size=2, stride=2)
        x1_2 = F.relu(self.conv1_2(x1_2))
        x1_4 = F.avg_pool2d(x1_2,kernel_size=2, stride=2)
        x1_4 = F.relu(self.conv1_3(x1_4))
        x1_2_ = F.upsample_nearest(x1_4, scale_factor=2)
        x1_2 = torch.cat([x1_2,x1_2_],1)
        x1_2 = F.relu(self.conv1_4(x1_2))
        x1_1_ = F.upsample_nearest(x1_2, scale_factor=2)
        x1_1 = torch.cat([x1_1,x1_1_],1)
        px = F.relu(self.conv1_5(x1_1))
        px = torch.cat([px,px,px],1)
        px = 1-px/16
        
        return px*x+(1-px)*avg

model = Net()
model.cuda()
model.train()

#define solver parameter, here we will have static parameter
lr = 0.01
momentum = 0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
losslayer = nn.MSELoss()
lossduringtraining = open("tmp/lossduringtraining.txt","w")

############ def train test

def tonumpy4D(image):
    #net expect 4D numpytensor
    #we do not use the first dim of this tensor which is designed for mini batch
    #notice that some layer requires mini batch of large size
    imagenp4 = np.zeros((1,3, imsize, imsize), dtype=float)
    imagenp3 = np.asarray(image)
    for ch in range(3):
        for r in range(imsize):
            for c in range(imsize):
                imagenp4[0][ch][r][c]=imagenp3[r][c][ch]/16
    return imagenp4

def trainbatch(inputnumpytensor,targetnumpytensor):
    #basic foward backward routine
    #data are converted to allow backward
    #then we do the forward
    #then we compute the loss
    #then we compute the gradient
    #finaly we update the weight
    
    data = torch.autograd.Variable(torch.Tensor(inputnumpytensor))
    target = torch.autograd.Variable(torch.Tensor(targetnumpytensor))
    data, target = data.cuda(), target.cuda()
    
    optimizer.zero_grad()
    output = model(data)
    
    loss = losslayer(output, target)
    lossduringtraining.write(str(loss.cpu().data.numpy()[0]*16*16)+"\n")
    lossduringtraining.flush()
    
    loss.cuda()
    loss.backward()
    
    optimizer.step()

def train():
    random.shuffle(traindata)
    for x,y in traindata :
        trainbatch(tonumpy4D(x),tonumpy4D(y))

def myminmax(v):
    if v<0:
        return 0
    if v*16>255:
        return 255
    return v*16
    
def testbatch(inputnumpytensor, inputindex):
    #here we only do the forward
    data = torch.autograd.Variable(torch.Tensor(inputnumpytensor),volatile=True)
    data = data.cuda()
    output = model(data).cpu().data.numpy()
    
    #then we get the output back into numpy3D and pil format
    impred = np.zeros((imsize, imsize,3), dtype=int)
    for r in range(imsize):
        for c in range(imsize):
            for ch in range(3):
                impred[r][c][ch] = myminmax(output[0][ch][r][c])
    predim = PIL.Image.fromarray(np.uint8(impred))
    predim.save("tmp/"+str(inputindex)+"z.png")
        
def test():
    for i in range(len(testdata)):
        x,y = testdata[i]
        testbatch(tonumpy4D(x),i)

print("start training testing")
for i in range(30):
    print(str(i))
    train()
test()
