"""CNN and small ViT-like vision baselines."""
import torch
from torch import nn
class TinyCNN(nn.Module):
 def __init__(self,num_classes=3):
  super().__init__()
  self.features=nn.Sequential(
   nn.Conv2d(3,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
   nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
   nn.Conv2d(32,48,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((2,3)))
  self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(48*2*3,64),nn.ReLU(),nn.Linear(64,num_classes))
 def forward(self,x):return self.classifier(self.features(x))
class TinyVisionTransformer(nn.Module):
 def __init__(self,num_classes=3,d_model=48,patch=16):
  super().__init__()
  self.patch=nn.Conv2d(3,d_model,kernel_size=patch,stride=patch)
  self.pos=nn.Parameter(torch.zeros(1,(64//patch)*(96//patch),d_model));nn.init.normal_(self.pos,std=.02)
  layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=4,dim_feedforward=96,dropout=0.,batch_first=True)
  self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False)
  self.head=nn.Linear(d_model,num_classes)
 def forward(self,x):
  tokens=self.patch(x).flatten(2).transpose(1,2)
  return self.head(self.encoder(tokens+self.pos).mean(dim=1))
def get_model(name):
 if name=='cnn':return TinyCNN()
 if name=='vit':return TinyVisionTransformer()
 raise ValueError('Unknown model')
