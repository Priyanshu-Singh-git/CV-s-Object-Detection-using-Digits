# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 22:55:30 2024
#line.strip(): Removes any leading/trailing whitespace, including the newline (\n),
# Splits the string into a list ['10', '20'] using space as the delimiter.
# map would apply int to the list
@author: Priyanshu singh
"""
import cv2
import numpy as np 
import json
import os 

import torch 
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random as rd

import matplotlib.pyplot as plt
 
from torchvision import datasets

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

trainset = datasets.MNIST(root="./data",train=True,transform=transform,download=True)

trainloader = DataLoader(dataset=trainset,batch_size=60,shuffle=True,num_workers=0)

def convert(img):
    #rd.seed(4)
    canvas = np.zeros((128,128))
    m = np.full((128,128), -1)
    canvas = canvas+m
    
    x,y = rd.randint(0, 100),rd.randint(0, 100)
    
    
    canvas[x:x+28,y:y+28] = img
    p1 = (y,x)
    print(p1)
    p2 = (y,x+28)
    print(p2)
    p3 = (y+28,x)
    print(p3)
    p4 = (y+28,x+28)
    print(p4)
    return canvas ,[p1,p2,p3,p4]

def show_bbox(img,p1,p4):#img type is numpy array
    img = (img+1)/2
    img = img*255
    
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    img = cv2.rectangle(img,p1,p4, (0,255,0),1)
    #img = cv2.resize(img,(512,512),interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow("",img)
    cv2.waitKey(1000)
    
#def save_img(img_path,bbox_path,idx):
    
    

def show_img(img):
    plt.imshow(img)
    plt.show()

#%%
cwd = os.getcwd()
#print(cwd)
root = "D:/pythonProject/Object Detection from Scratch"
os.chdir(root)
#print(cwd)

for i in os.listdir(cwd):
    print(i)
    
if not os.path.exists("D:/pythonProject/Object Detection from Scratch/train"):
    os.makedirs(os.path.join(root, "train/Images"))
    os.makedirs(os.path.join(root, "train/label"))
    



#%%
j = 0
for i,(datas,labels) in enumerate(trainloader):
    print(i)
    for data in datas:
        data  = data.squeeze(0).detach().numpy()
        cnvrtimg , bbox = convert(data)
        show_bbox(cnvrtimg,bbox[0],bbox[3])
        #print(cnvrtimg.shape)
        cnvrtimg = (cnvrtimg+1)/2
        cnvrtimg = cnvrtimg*255
        cnvrtimg = cnvrtimg.astype(np.uint8)
        print("j",j)
        if os.path.exists(imgpth:=os.path.join(root,"train/Images")) and os.path.exists(bboxpth:=os.path.join(root, "train/label")):
            img_name = os.path.join(imgpth, f"{j}.png")
            bbox_name = os.path.join(bboxpth,f"{j}.txt")
        
            cv2.imwrite(img_name, cnvrtimg)
        
            with open(bbox_name,"w") as f :
            
                for coor in bbox:
                    f.write(f"{coor[0]} {coor[1]}\n")
        j+=1
                    

    

#%%
from torch.utils.data import DataLoader,Dataset
from PIL import Image

class Custom_trainDataset(Dataset):
    def __init__(self,transform = None):
        self.label_dir = os.path.join(root,"train/label")
        self.target_dir = os.path.join(root,"train/Images")
        self.transform = transform
        
        self.labels = sorted(os.listdir(self.label_dir))
        self.targets = sorted(os.listdir(self.target_dir))
        
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        target_file = os.path.join(self.target_dir,self.targets[idx])
        label_file = os.path.join(self.label_dir,self.labels[idx])
        bbox = []
        
        target_img = Image.open(target_file).convert("L")
        
        
            
            
        with open(label_file,"r") as file:
            for coor in file:
                x,y = map(int,coor.strip().split()) 
                bbox.append((x,y))
        if self.transform:
            target_img = self.transform(target_img)
            bbox = torch.tensor(bbox)
        #print(bbox)
    
                
        return target_img,bbox
transform2 = transforms.Compose(
            [
                transforms.ToTensor(),
                ])

custom_trainset = Custom_trainDataset(transform=transform2)
train_dataloader = DataLoader(dataset=custom_trainset,shuffle=True,batch_size = 20)
#%%
for data,bbox in train_dataloader:
    data = data[0].cpu().numpy().transpose((1,2,0))
    print(type(bbox))
    print(len(bbox))
    print(len(bbox[0]))
    print(len(bbox[0][0]))
    
    print(tuple(bbox[0][0].tolist()),tuple(bbox[0][3].tolist()))
    print('---------------------------------------------------------------------')
    
    show_bbox(data,tuple(bbox[0][0].tolist()),tuple(bbox[0][3].tolist()))
    
        
        
        
        
        
#%%
class Obj_dec(nn.Module):
        def __init__(self):
            super(Obj_dec, self).__init__()
            
            self.main = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size=5,padding=0,stride=1,bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=1,padding=0),
                
                nn.Conv2d(in_channels = 64, out_channels = 64*2, kernel_size=5,padding=0,stride=1,bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                nn.MaxPool2d(kernel_size=3,stride=1,padding=0),
                
                
                nn.Conv2d(in_channels = 64*2, out_channels = 64*4, kernel_size=5,padding=0,stride=1,bias=False),
                nn.BatchNorm2d(64*4),
                nn.LeakyReLU(0.2,inplace = True),
                nn.Dropout2d(0.3),
                nn.Conv2d(in_channels=64*4, out_channels=64*4, kernel_size=5,padding = 0,stride = 1,bias=False),
                nn.BatchNorm2d(64*4),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Dropout2d(0.3),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
                
                nn.Conv2d(in_channels = 64*4, out_channels = 64*8, kernel_size=3,padding=1,stride=1,bias=False),
                nn.BatchNorm2d(64*8),
                nn.LeakyReLU(0.2,inplace = True),
                nn.Dropout2d(0.35),
                nn.Conv2d(in_channels=64*8, out_channels=64*8, kernel_size=3,padding = 1,stride = 1,bias=False),
                nn.BatchNorm2d(64*8),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Dropout2d(0.35),
                nn.MaxPool2d(kernel_size=2,stride=4,padding=1),
                 
                 
                nn.Conv2d(in_channels = 64*8, out_channels = 64*16, kernel_size=3,padding=1,stride=1,bias=False),
                nn.BatchNorm2d(64*16),
                
                nn.LeakyReLU(0.2,inplace = True),
                nn.Dropout2d(0.425),
                  
                 
                nn.Conv2d(in_channels=64*16, out_channels=64*16, kernel_size=3,padding = 1,stride = 1,bias=False),
                nn.BatchNorm2d(64*16),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Dropout2d(0.45),
                nn.MaxPool2d(kernel_size=2,stride=4,padding=1),
                nn.Flatten(),
                nn.Linear(16384,1000,bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                
                nn.Linear(1000,8,bias=True),
                nn.LeakyReLU(0.2,inplace=True),
                
                
                
                )
        def forward(self,input):
            return self.main(input)
        



print(Obj_dec())
device = torch.device("cuda")
model = Obj_dec().to(device)

for name,param in model.named_parameters():
    print(name)
    
    
pretrained_weights = torch.load("D:/pythonProject/MNIST_ANN_CNN/cnn_digit_class_final.pth",map_location=device)

for names in pretrained_weights.keys():
    print(names)
pretrained_weights = {k: v for k, v in pretrained_weights.items() if "main.34" not in k and "main.36" not in k}


model.load_state_dict(pretrained_weights,strict=False)
    
def show_compare_bbox(img,output,bbox):#img type is numpy array
    img = (img+1)/2
    img = img*255
    
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    
    p1 = (int(bbox[0]),int(bbox[1]))
    p4 = (int(bbox[6]),int(bbox[7]))
    img = cv2.rectangle(img,p1,p4, (0,255,0),1)
    
    o1 = (int(output[0]),int(output[1]))
    o4 = (int(output[6]),int(output[7]))
    img = cv2.rectangle(img,o1,o4, (0,0,255),1)
    cv2.imshow("",img)
    cv2.waitKey(100)
    
def show_compare_bbox2(img, output, bbox):  # img type is numpy array
    img = (img + 1) / 2
    img = img * 255
    img = img.astype(np.uint8)

    # If the image is grayscale with a single channel, convert it to RGB
    if len(img.shape) == 2:  # Single channel grayscale image
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 1:  # Single channel grayscale image with shape (H, W, 1)
        img = np.squeeze(img, axis=-1)  # Remove the singleton channel dimension
        img = np.stack([img] * 3, axis=-1)

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # Draw the ground truth bbox in green
    p1 = (bbox[0], bbox[1])
    width = bbox[6] - bbox[0]
    height = bbox[7] - bbox[1]
    rect1 = plt.Rectangle(p1, width, height, edgecolor='green', facecolor='none', linewidth=1)
    ax.add_patch(rect1)

    # Draw the predicted bbox in red
    o1 = (output[0], output[1])
    width = output[6] - output[0]
    height = output[7] - output[1]
    rect2 = plt.Rectangle(o1, width, height, edgecolor='red', facecolor='none', linewidth=1)
    ax.add_patch(rect2)

    # Display the image
    plt.show()


criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i,(data,bbox) in enumerate(train_dataloader):
        
        data = data.to(device)
        optimizer.zero_grad()
        
        output = model(data).to(device)
        
        bbox = bbox.view(bbox.size(0),-1).to(device)
        output = output.view(output.size(0),-1).to(device)
        
        loss = criterion(output,bbox)
        loss.backward()
        
        optimizer.step()
        epoch_loss +=loss.item()
        
        
        
        img = data[0].cpu().numpy().transpose((1,2,0))
        bbox_list = bbox[0].cpu().tolist()
        output_list = output[0].cpu().tolist()
        show_compare_bbox2(img, output_list, bbox_list)
        print(bbox[0],"-----------" , output[0])
        print(f"Batch no : {i} ------------- Batch_loss :{loss.item()}")
    print(f"Epoch : {epoch}----------------- Epoch_loss :{epoch_loss}")
    
#%%

torch.save(model.state_dict(),"D:/pythonProject/Object Detection from Scratch/objdec.pth")

    
    


