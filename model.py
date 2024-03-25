import torch.nn as nn 
import torch

class ImageClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.3)
        self.last_layer = nn.Linear(2048, 10)
        #self.last_layer1 = nn.Linear(11, 11)
        
    def forward(self, x):
        o = self.backbone(x)
        o = self.dropout(o)
        o = o.squeeze()
        o = self.last_layer(o)
        #o = self.last_layer1(o)
        return o



