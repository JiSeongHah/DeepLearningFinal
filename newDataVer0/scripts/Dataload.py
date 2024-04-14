import os
import yaml
from pprint import pprint
from torchvision import datasets, transforms
import torch


def loadData(configs, isTrain):

    dataDict = {
        "mnist",
        "cifar10",
        "mvtecad",
        "stl10",
        "crwu",
    }

    whichData = configs["data_type"]
    dataDownPath = os.path.join(configs["data_down_path"], whichData)

    os.makedirs(dataDownPath, exist_ok=True)

    if whichData == "mnist":
        loadedData = datasets.MNIST(
            root=dataDownPath,
            train=isTrain,
            download=True,
            transform=transforms.ToTensor(),
        )

        return loadedData

    elif whichData == "cifar10":
        loadedData = datasets.CIFAR10(
            root=dataDownPath,
            train=isTrain,
            download=True,
            transform=transforms.ToTensor(),
        )

        return loadedData

    else:
        pass


class myNewDataset(torch.utils.data.Dataset):
    def __init__(self, configs, isTrain):

        self.configs = configs
        self.isTrain = isTrain
        self.doFlatten = configs["do_flatten"]

        self.loadedData = loadData(configs=configs, isTrain=isTrain)

    def __len__(self):
        return len(self.loadedData)

    def __getitem__(self, idx):

        data = self.loadedData[idx][0]

        if self.doFlatten:
            data = torch.flatten(data)

        label = self.loadedData[idx][1]

        return data, label




# yamlPath = '.configs/config.yaml'

# loadedYaml = readConfig(yamlPath)
# print(os.getcwd())
# lst = os.listdir(os.getcwd())

# for i in lst:
#     print(i)
# testDataset = myNewDataset(configs=loadedYaml,isTrain=True)
# import time
# for i in testDataset:
#     print(i[0].size(),i[1])
#     time.sleep(1)
