from turtle import back
from sklearn import model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, densenet121, resnext50_32x4d, squeezenet1_0, inception_v3, densenet161

class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, squeeze=False):
        super().__init__()
        if squeeze:
            self.fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Linear(in_dim, out_dim, bias=False),
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x
        
class Model(nn.Module):
    
    def __init__(self, class_num,  model_name='resnet50', finetune=False):
        # in_dim = feature_dim, out_dim = class_num
        super().__init__()
        
        if model_name == 'resnet':
            in_dim = 2048
            squeeze = False
            backbone = resnet50(pretrained=True)

        if model_name == 'resnetxt':
            in_dim = 2048
            squeeze = False
            backbone = resnext50_32x4d(pretrained=True)

        if model_name == 'densenet':
            in_dim = 1024
            backbone = densenet121(pretrained=False)

            classifier = Classifier(in_dim, class_num)
            self.backbone = backbone
            self.backbone.classifier = classifier

        if model_name == 'densenet_max':
            in_dim = 1024
            squeeze = True
            backbone = densenet161(pretrained=True)

            classifier = Classifier(in_dim, class_num)
            self.backbone = backbone
            self.backbone.classifier = classifier

        if model_name == 'inception':
            in_dim = 2048
            backbone = inception_v3(pretrained=True)

            classifier = Classifier(in_dim, class_num)
            self.backbone = backbone
            self.backbone.fc = classifier

        self.model_name = model_name
        self.finetune = finetune

        #self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        #self.classifier = Classifier(in_dim, class_num, squeeze)
        #print(self.backbone)

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        
    def forward(self, x):

        x = self.backbone(x)
        if len(x) == 2:
            x = x[0]
        return x

    def get_parameters(self, base_lr=1.0):

        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.classifier.parameters(), "lr": 1.0 * base_lr},
        ]

        return params
