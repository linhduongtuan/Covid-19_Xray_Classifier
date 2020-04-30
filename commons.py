import io
import PIL
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from geffnet import create_model
import copy
from copy import deepcopy
from torch import optim
from collections import OrderedDict
from timm.models.layers.activations import *


def get_tensor(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_model():
    model = create_model('efficientnet_b0', pretrained=True)
    for param in model.parameters():
        param.requires_grad = True
    
    fc = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 1000, bias=True)),
                                    ('BN1', nn.BatchNorm2d(1000, eps=1e-05, momentum=0.1, affine=True,
                                                           track_running_stats=True)),
                                    ('dropout1', nn.Dropout(0.7)),
                                    ('fc2', nn.Linear(1000, 512)),
                                    ('BN2', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True,
                                                           track_running_stats=True)),
                                    ('swish1', Swish()),
                                    ('dropout2', nn.Dropout(0.5)),
                                    ('fc3', nn.Linear(512, 128)),
                                    ('BN3', nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True,
                                                           track_running_stats=True)),
                                    ('swish2', Swish()),
                                    ('fc4', nn.Linear(128, 3)),
                                    ('output', nn.Softmax(dim=1))
                                    ]))
    model.fc = fc
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    CHECK_POINT_PATH = '/Users/mac/PycharmProjects/Covid-19_Chest_Xray/weights/EfficientNet_B0_Covid-19.pth'
    
    checkpoint = torch.load(CHECK_POINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    best_model_wts = copy.deepcopy(model.state_dict())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_loss = checkpoint['best_val_loss']
    best_acc = checkpoint['best_val_accuracy']
    model.eval()
    return model


model = get_model()
model.eval()

