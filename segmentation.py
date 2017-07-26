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
imsize=512
testdata = []
testdata.append((PIL.Image.open("data/image.jpg").copy(),PIL.Image.open("data/mask.jpg").convert("RGB").copy()))
    
#this is a non sense from a learning point of view, but here, we just learn on the testing data for simplicity 
traindata = testdata.copy()

#create a tmp folder to store prediction
os.makedirs("tmp")
for x,y in testdata:
    x.save("tmp/x.jpg")
    y.save("tmp/y.jpg")

#here we define a network for this segmentation problem with downscaled output
#in this case we just use a vgg
#note that writing the network is not a big deal
#the problem is to find the one on which optimization will behave correctly and/or for whose there are pre trained model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1_1 =    nn.Conv2d(3, 64, kernel_size=3,padding=1, bias=True)
        self.conv1_2 =   nn.Conv2d(64, 64, kernel_size=3,padding=1, bias=True)
        self.conv2_1 =  nn.Conv2d(64, 128, kernel_size=3,padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3,padding=1, bias=True)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3,padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3,padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3,padding=1, bias=True)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3,padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,padding=1, bias=True)
        
        self.prob1 = nn.Conv2d(64, 2, kernel_size=1, bias=True)
        self.prob2 = nn.Conv2d(128, 2, kernel_size=1, bias=True)
        self.prob4 = nn.Conv2d(256, 2, kernel_size=1, bias=True)
        self.prob8 = nn.Conv2d(512, 2, kernel_size=1, bias=True)
        self.prob16 = nn.Conv2d(512, 2, kernel_size=1, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        p1 = self.prob1(x)
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        p2 = self.prob2(x)
        p2=F.upsample_nearest(p2, scale_factor=2)
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        p4 = self.prob4(x)
        p4=F.upsample_nearest(p4, scale_factor=4)
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        p8 = self.prob8(x)
        p8=F.upsample_nearest(p8, scale_factor=8)
        x = F.max_pool2d(x,kernel_size=2, stride=2,return_indices=False)
        
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        p16 = self.prob16(x)
        p16=F.upsample_nearest(p16, scale_factor=16)

        return p1/16+p2/8+p4/4+p8/2+p16
        
    def load_weights(self, model_path):
        correspondance=[]
        correspondance.append(("features.0","conv1_1"))
        correspondance.append(("features.2","conv1_2"))
        correspondance.append(("features.5","conv2_1"))
        correspondance.append(("features.7","conv2_2"))
        correspondance.append(("features.10","conv3_1"))
        correspondance.append(("features.12","conv3_2"))
        correspondance.append(("features.14","conv3_3"))
        correspondance.append(("features.17","conv4_1"))
        correspondance.append(("features.19","conv4_2"))
        correspondance.append(("features.21","conv4_3"))
        correspondance.append(("features.24","conv5_1"))
        correspondance.append(("features.26","conv5_2"))
        correspondance.append(("features.28","conv5_3"))
        
        model_dict = self.state_dict()
        pretrained_dict = torch.load(model_path)    
        
        for name1,name2 in correspondance:
            fw = False
            fb = False
            for name, param in pretrained_dict.items():
                if name==name1+".weight" :
                    model_dict[name2+".weight"].copy_(param)
                    fw=True
                if name==name1+".bias" :
                    model_dict[name2+".bias"].copy_(param)
                    fb=True
            if not fw:
                print(name2+".weight not found")
            if not fb:
                print(name2+".bias not found")
        self.load_state_dict(model_dict)
        
vgg = VGG()
vgg.load_weights("data/vgg16-00b39a1b.pth")
vgg.cuda()
vgg.train()

#define solver parameter, here we will have static parameter
lr = 0.0001
momentum = 0.5
optimizer = optim.SGD(vgg.parameters(), lr=lr, momentum=momentum)
losslayer = nn.CrossEntropyLoss()
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
                imagenp4[0][ch][r][c]=imagenp3[r][c][ch]
    return imagenp4
    
def tonumpy4Dmask(mask):
    imagenp4 = np.zeros((1,1, imsize, imsize), dtype=int)
    imagenp3 = np.asarray(mask)
    for r in range(imsize):
        for c in range(imsize):
            imagenp4[0][0][r][c]=(imagenp3[r][c][0]!=0)
    return imagenp4

nbepoch = 60

print("start training testing")
for epoch in range(nbepoch):
    print(str(epoch))
    for x,y in testdata:
        inputtensor = torch.autograd.Variable(torch.Tensor(tonumpy4D(x)))
        inputtensor = inputtensor.cuda()
        optimizer.zero_grad()
        
        outputtensor = vgg(inputtensor)
        
        targettensor = torch.from_numpy(tonumpy4Dmask(y))
        targettensor = torch.autograd.Variable(targettensor)
        targettensor = targettensor.cuda()
        
        outputtensor = outputtensor.view(outputtensor.size(0),outputtensor.size(1), -1)
        outputtensor = torch.transpose(outputtensor,1,2).contiguous()
        outputtensor = outputtensor.view(-1,outputtensor.size(2))
        targettensor = targettensor.view(-1)
        loss = losslayer(outputtensor, targettensor)
        loss.backward()
        optimizer.step()
        
        lossduringtraining.write(str(loss.cpu().data.numpy()[0])+"\n")
        lossduringtraining.flush()

print("testing")
for x,y in testdata:
    inputtensor = torch.autograd.Variable(torch.Tensor(tonumpy4D(x)))
    inputtensor = inputtensor.cuda()
    optimizer.zero_grad()
    
    outputtensor = vgg(inputtensor)
    
    proba = outputtensor.cpu().data.numpy()
    impred = np.zeros((imsize, imsize,3), dtype=int)
    for r in range(imsize):
        for c in range(imsize):
            for ch in range(3):
                impred[r][c][ch] = 255*((proba[0][1][r][c]-proba[0][0][r][c])>0)
    predim = PIL.Image.fromarray(np.uint8(impred))
    predim.save("tmp/z.jpg")
    
