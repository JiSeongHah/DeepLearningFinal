import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class naiveFCN(nn.Module):
    def __init__(self, hDim1, hDim2, hDim3, FVSize, inputSize=56):
        super(naiveFCN, self).__init__()

        self.inputSize = inputSize

        self.hDim1 = hDim1

        self.hDim2 = hDim2

        self.hDim3 = hDim3

        self.FVSize = FVSize

        self.lin1 = nn.Linear(
            in_features=self.inputSize, out_features=self.hDim1, bias=False
        )
        self.lin2 = nn.Linear(
            in_features=self.hDim1, out_features=self.hDim2, bias=False
        )
        self.lin3 = nn.Linear(
            in_features=self.hDim2, out_features=self.hDim3, bias=False
        )
        self.lin4 = nn.Linear(
            in_features=self.hDim3, out_features=self.FVSize, bias=False
        )

    def forward(self, x):

        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = self.lin4(out)

        return out


class naivePreAutoEncoder(nn.Module):
    def __init__(self, hDim1, hDim2, hDim3, FVSize, inputSize=56):
        super(naivePreAutoEncoder, self).__init__()

        self.inputSize = inputSize

        self.hDim1 = hDim1

        self.hDim2 = hDim2

        self.hDim3 = hDim3

        self.FVSize = FVSize

        self.lin1 = nn.Linear(
            in_features=self.inputSize, out_features=self.hDim1, bias=False
        )
        self.lin2 = nn.Linear(
            in_features=self.hDim1, out_features=self.hDim2, bias=False
        )
        self.lin3 = nn.Linear(
            in_features=self.hDim2, out_features=self.hDim3, bias=False
        )
        self.lin4 = nn.Linear(
            in_features=self.hDim3, out_features=self.FVSize, bias=False
        )
        self.lin5 = nn.Linear(
            in_features=self.FVSize, out_features=self.hDim3, bias=False
        )
        self.lin6 = nn.Linear(
            in_features=self.hDim3, out_features=self.hDim2, bias=False
        )
        self.lin7 = nn.Linear(
            in_features=self.hDim2, out_features=self.hDim1, bias=False
        )
        self.lin8 = nn.Linear(
            in_features=self.hDim1, out_features=self.inputSize, bias=False
        )

    def doEncode(self, x):

        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = self.lin4(out)

        return out

    def doDecode(self, x):

        out = F.relu(self.lin5(F.relu(x)))
        out = F.relu(self.lin6(out))
        out = F.relu(self.lin7(out))
        out = self.lin8(out)

        return out

    def forward(self, x):

        out = self.doEncode(x)
        out = self.doDecode(out)

        return out
