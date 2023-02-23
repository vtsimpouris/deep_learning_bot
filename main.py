from pynput.mouse import Listener
import numpy as np
import cv2
from PIL import ImageGrab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pyautogui
from pyautogui import *
from pynput.keyboard import Key, Controller
from pynput.mouse import Button, Controller as MController
import threading
import random
import math


class callback_store:
    x = None
    Y = None
    def on_press(key):
        print(key)

def hit():
   while(True):
        d_q = random.randint(0, 5)
        s_a = random.randint(0, 5)
        shield = random.randint(0, 5)
        if (d_q == 4):
            keyboard.press('1')
            keyboard.release('1')
        if (shield == 4):
            keyboard.press('3')
            keyboard.release('3')
        if (s_a == 4):
            keyboard.press('2')
            time.sleep(5)
            keyboard.release('2')
        for j in range(4):
            keyboard.press('b')
            freq = random.uniform(0.2, 0.4)
            time.sleep(freq)
            keyboard.release('b')
            keyboard.press('q')
            time.sleep(freq)
            keyboard.release('q')
            keyboard.press('4')
            time.sleep(freq)
            keyboard.release('4')

def generator():
    cs = callback_store()
    #while(1):
    with Listener(on_press=cs.on_press) as listener:
            listener.join()
            img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))  # bbox specifies specific region (bbox= x,y,width,height)
            img = np.array(img)
            return cs.x, cs.y, img
            #cv2.imshow('s', img_np)
            #cv2.waitKey(0)

def endless_generator():
        img = ImageGrab.grab(bbox=(0, 0, 1920, 1080))  # bbox specifies specific region (bbox= x,y,width,height)
        img = np.array(img)
        return img
        #cv2.imshow('s', img_np)
        #cv2.waitKey(0)



class myCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
       #self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        #self.fc1 = nn.Linear(5120, 2048)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)

        self.num_classes = 2 #number of classes
        self.num_layers = 1 #number of layers
        self.input_size = 256 #input size
        self.hidden_size = 128 #hidden state
        self.seq_length = 10 #sequence length

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers) #lstm



    def forward(self, xb):
        xb = xb.view(-1, 3, 1080, 1300)
        xb = F.relu(self.conv1(xb))
        xb = F.avg_pool2d(xb, 4)
        xb = F.relu(self.conv2(xb))
        xb = F.avg_pool2d(xb, 4)
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)
        #xb = F.relu(self.conv4(xb))
        #xb = F.relu(self.conv7(xb))
        #xb = F.avg_pool2d(xb, 4)
        xb = xb.view(xb.size(0), -1)
        output, (hidden, _) = self.lstm(xb.float())
        print(xb.shape)
        #xb = self.fc1(xb)
        #xb = self.fc2(xb)
        #xb = self.fc3(xb)
        #xb = F.softmax(xb, dim=1)
        #xb = torch.sigmoid(self.fc3(xb))
        return output

num_epochs = 8
num_classes = 2
batch_size = 1
learning_rate = 0.01

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = myCNN().to(device)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)

keyboard = Controller()
trainable = True

if(trainable):
    for i in range(100):
        #print(i)
        x,y,img = generator()
        #cv2.imshow('CHAIN_APPROX_SIMPLE Point only', img)
        #cv2.waitKey(0)
        #exit(1)
        #print(x)
        #print(y)
        #print(img.shape)
        images = torch.from_numpy(img).cuda(0)
        images = images.to(torch.float)
        xy = np.array([x,y])
        coords = torch.from_numpy(xy).cuda(0)
        coords = coords.to(torch.float)
        #print(coords)
        #images = img.to(device)
        #labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        #print(outputs)
        #print(xy)
        coords = coords[None,:]
        loss = criterion(outputs, coords)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(math.sqrt(loss.item()))
    #torch.save(model, "C:/Users/SKIKK/PycharmProjects/pytorch_example/img_id.py")
else:
    model = torch.load("C:/Users/SKIKK/PycharmProjects/pytorch_example/img_id.py")
    t1 = threading.Thread(target=hit, args=(), daemon=True)
    t1.start()


pyautogui.FAILSAFE = False
mouse = MController()

while(1):
    img = endless_generator()
    time.sleep(0.2)
    images = torch.from_numpy(img).cuda(0)
    images = images.to(torch.float)
    #xy = np.array([x,y])
    #coords = torch.from_numpy(xy).cuda(0)
    #coords = coords.to(torch.float)
    #print(coords)
    #images = img.to(device)
    #labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    outputs = outputs.cpu().detach().numpy()
    #print(outputs[0][0],outputs[0][1])
    moveTo(outputs[0][0],outputs[0][1])  # monastery
    for s in range(5):
        time.sleep(0.1)
        for kl in range(10):
            mouse.click(Button.right, 1)
            time.sleep(0.01)

