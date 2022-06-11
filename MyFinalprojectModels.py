import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torchvision import models
import math
import timm



class simpleDNN(nn.Module):

    def __init__(self,
                 innerNum1=64,
                 innerNum2=64,
                 innerNum3=64):
        super(simpleDNN, self).__init__()


        self.innerNum1 = innerNum1
        self.innerNum2 = innerNum2
        self.innerNum3 = innerNum3

        self.lin1 = nn.Linear(in_features=17,out_features=self.innerNum1)
        self.lin2 = nn.Linear(in_features=self.innerNum1,out_features=self.innerNum2)
        self.lin3 = nn.Linear(in_features=self.innerNum2, out_features=self.innerNum3)
        self.lin4 = nn.Linear(in_features=self.innerNum3, out_features=1)

    def forward(self, x):
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = self.lin4(out)

        return out


# model =simpleDNN()
#
# x = torch.randn(3,17)
#
# y = model(x)
# z = torch.zeros_like(y)
# loss = torch.sum((y-z)**2)
# print(model.lin1.weight.grad)
# loss.backward()
#
# print(model.lin1.weight.grad)
#

x =torch.tensor(1)
print(torch.sqrt(1.0-x))