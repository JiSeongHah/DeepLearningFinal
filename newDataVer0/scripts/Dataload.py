import os
import yaml
from pprint import pprint
from torchvision import datasets, transforms
import torch
import pickle
from joblib import dump, load

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
    
    testBed_dataLst = [f'800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]
    
    FEed_testBed_dataLst = [f'FEed_800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'FEed_800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]
    
    
    testBest_intervalLst= []
    intervalLst = [128,256,512,1024,2048,4096]
    
    for interval in intervalLst:
        testBest_intervalLst.extend([f'800_0_interval_{interval}_noiseRatio_{round(0.1*i,1)}' for i in range(10)])
        testBest_intervalLst.extend([f'800_45_interval_{interval}_noiseRatio_{round(0.1*i,1)}' for i in range(10)])
        
    if whichData == "mnist":
        loadedData = datasets.MNIST(
            root=dataDownPath,
            train=isTrain,
            download=True,
            transform=transforms.ToTensor(),
        )

        return loadedData

    elif whichData in [f"mnist_{i}" for i in range(1, 11)]:

        if isTrain is True:
            noise_intensity = whichData.split("_")[1]
            dataLoadPath = f"/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/data_download_path/noisedData/mnist/train/noise_{noise_intensity}.pickle"

            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)

        if isTrain is False:
            noise_intensity = whichData.split("_")[1]
            dataLoadPath = f"/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/data_download_path/noisedData/mnist/test/noise_{noise_intensity}.pickle"

            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)

        return loadedData

    elif whichData == "cifar10":
        loadedData = datasets.CIFAR10(
            root=dataDownPath,
            train=isTrain,
            download=True,
            transform=transforms.ToTensor(),
        )

        return loadedData

    elif whichData in [f"cifar_{i}" for i in range(1, 11)]:

        if isTrain is True:
            noise_intensity = whichData.split("_")[1]
            dataLoadPath = f"/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/data_download_path/noisedData/cifar/train/noise_{noise_intensity}.pickle"

            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)

        if isTrain is False:
            noise_intensity = whichData.split("_")[1]
            dataLoadPath = f"/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/data_download_path/noisedData/cifar/test/noise_{noise_intensity}.pickle"

            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)

        return loadedData

    
    elif whichData in testBed_dataLst:
        
        dataName = whichData.split("_noiseRatio_")[0]
        actualNoise = whichData.split("noiseRatio_")[-1]
        
        if isTrain is True:
            
            dataLoadPath = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/noised_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}.pkl'
            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)
                
            print(loadedData)
                
        if isTrain is False:
            
            dataLoadPath = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/noised_data/{whichData}/test_{dataName}_noised_{actualNoise}.pkl'
            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)
            
        return loadedData
    
    elif whichData in FEed_testBed_dataLst:
        
        dataName = whichData.split("_noiseRatio_")[0].split('ed_')[-1]
        actualNoise = whichData.split("noiseRatio_")[-1]
        
        if isTrain is True:
            
            dataLoadPath = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/FEed_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}.pkl'
            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)
                
        if isTrain is False:
            
            dataLoadPath = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/FEed_data/{whichData}/test_{dataName}_noised_{actualNoise}.pkl'
            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)
                
                
    elif whichData in testBest_intervalLst:
        
        tmpBaseDir = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/noised_data/'
        
        dataName = whichData.split("_noiseRatio_")[0]
        actualNoise = whichData.split("_noiseRatio_")[-1]
        
        if isTrain is True:
            
            dataLoadPath = os.path.join(
                tmpBaseDir,
                f'{whichData}/trainVal_{dataName}_noised_{actualNoise}.pkl'
                )
            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)
                
        if isTrain is False:
            
            dataLoadPath = os.path.join(
                tmpBaseDir,
                f'{whichData}/test_{dataName}_noised_{actualNoise}.pkl'
            )
            
            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)
        
        return loadedData

def openPickle(loadPath):
    
    with open(loadPath, 'rb') as f:
        data = pickle.load(f)
        
        return data

class myNewDataset(torch.utils.data.Dataset):
    def __init__(self, configs, isTrain):

        self.configs = configs
        self.isTrain = isTrain
        self.doFlatten = configs["do_flatten"]
        
        self.dataType = configs["data_type"]
        
        self.loadedData = loadData(configs=configs, isTrain=isTrain)
        
        self.noised_data_lst = [f'800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]
        self.FEed_data_lst = [f'FEed_800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'FEed_800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]
        
        self.testBest_intervalLst= []
        intervalLst = [128,256,512,1024,2048,4096]
        for interval in intervalLst:
            self.testBest_intervalLst.extend([f'800_0_interval_{interval}_noiseRatio_{round(0.1*i,1)}' for i in range(10)])
            self.testBest_intervalLst.extend([f'800_45_interval_{interval}_noiseRatio_{round(0.1*i,1)}' for i in range(10)])
            
        if self.dataType in self.testBest_intervalLst:
            
            whichData = configs["data_type"]
            dataName = whichData.split("_noiseRatio_")[0]
            actualNoise = whichData.split("noiseRatio_")[-1]
        
            self.mean_std_dict = openPickle(
                f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/noised_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}_mean_std.pkl'
            )
            
        if self.dataType in self.noised_data_lst:
            whichData = configs["data_type"]
            dataName = whichData.split("_noiseRatio_")[0]
            actualNoise = whichData.split("noiseRatio_")[-1]
        
            self.mean_std_dict = openPickle(
                f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/noised_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}_mean_std.pkl'
            )
        # elif self.dataType in FEed_data_lst:
            
        #     whichData = configs["data_type"]
        #     dataName = whichData.split("_noiseRatio_")[0]
        #     actualNoise = whichData.split("noiseRatio_")[-1]
            
        #     self.scaler = load(
        #         f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/FEed_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}_scaler.joblib'
        #     )

    def __len__(self):
        return len(self.loadedData)

    def __getitem__(self, idx):
        
        
        
        
        if self.dataType in [f'800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]:
            
            data = self.loadedData[idx][0]
            self.doZScore = self.configs['do_zScore']
            if self.doZScore:
                print('tesssssssssssssssssssssssssssssssssssssssssss')
                meanValue = self.mean_std_dict['mean']
                stdValue = self.mean_std_dict['std']
                
                data = (data-meanValue)*(1/(stdValue+1e-9))
            

            # if self.doFlatten:
            #     data = torch.flatten(data)

            label = self.loadedData[idx][1]
            
            flg = self.loadedData[idx][2]

            return data, label, flg
        
        elif self.dataType in [f'FEed_800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'FEed_800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]:
            
            
            data = self.loadedData[idx][0]
            
            # if self.doZScore:
            #     print('tesssssssssssssssssssssssssssssssssssssssssss')
            #     meanValue = self.mean_std_dict['mean']
            #     stdValue = self.mean_std_dict['std']
                
            #     data = (data-meanValue)*(1/(stdValue+1e-9))

            # if self.doFlatten:
            #     data = torch.flatten(data)

            label = self.loadedData[idx][1]
            
            flg = self.loadedData[idx][2]

            return data, label, flg
            
        
        elif self.dataType in self.testBest_intervalLst:
            
            data = self.loadedData[idx][0]
            self.doZScore = self.configs['do_zScore']
            if self.doZScore:
                
                meanValue = self.mean_std_dict['mean']
                stdValue = self.mean_std_dict['std']
                
                data = (data-meanValue)*(1/(stdValue+1e-9))
            
            label = self.loadedData[idx][1]
            
            flg = self.loadedData[idx][2]

            return data, label, flg
            
            
        else:

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
