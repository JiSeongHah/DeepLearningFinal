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
    
    FEed_testBed_interval_dataLst = []
    datatype_lst = ['800_0','800_45']
    interval_lst = [128,256,512,1024,2048,4096]
    noise_ratio_lst = [round(0.1*i,1) for i in range(10)]
    for datatype_1 in datatype_lst:
        for interval_1 in interval_lst:
            for nonoise_ratio in noise_ratio_lst:
                FEed_testBed_interval_dataLst.append(f'FEed_{datatype_1}_interval_{interval_1}_noiseRatio_{nonoise_ratio}')
    
    
    testBest_intervalLst= []
    intervalLst = [128,256,512,1024,2048,4096]
    
    for interval in intervalLst:
        testBest_intervalLst.extend([f'800_0_interval_{interval}_noiseRatio_{round(0.1*i,1)}' for i in range(10)])
        testBest_intervalLst.extend([f'800_45_interval_{interval}_noiseRatio_{round(0.1*i,1)}' for i in range(10)])
        
    if whichData == "mnist_yes":
        
        if isTrain is True:
            
            with open(
                '/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/data_download_path/non_noise_data/mnist_ver/mnist_processed_train.pkl',
                'rb'
            ) as F:
                
                loadedData = pickle.load(F)
        else:
            with open(
                '/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/data_download_path/non_noise_data/mnist_ver/mnist_processed_test.pkl',
                'rb'
            ) as F:
                
                loadedData = pickle.load(F)

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

    elif whichData == "cifar_yes":
        
        if isTrain is True:
            
            with open(
                '/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/data_download_path/non_noise_data/cifar_ver/cifar_processed_train.pkl',
                'rb'
            ) as F:
                
                loadedData = pickle.load(F)
        else:
            with open(
                '/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/data_download_path/non_noise_data/cifar_ver/cifar_processed_test.pkl',
                'rb'
            ) as F:
                
                loadedData = pickle.load(F)
        

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
        
        raise Exception('legacy error 1')
        
        # dataName = whichData.split("_noiseRatio_")[0]
        # actualNoise = whichData.split("noiseRatio_")[-1]
        
        # if isTrain is True:
            
        #     dataLoadPath = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/noised_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}.pkl'
        #     with open(dataLoadPath, "rb") as F:
        #         loadedData = pickle.load(F)
                
        #     print(loadedData)
                
        # if isTrain is False:
            
        #     dataLoadPath = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/noised_data/{whichData}/test_{dataName}_noised_{actualNoise}.pkl'
        #     with open(dataLoadPath, "rb") as F:
        #         loadedData = pickle.load(F)
            
        # return loadedData
    
    elif whichData in FEed_testBed_dataLst:
        
        raise Exception('legacy error 2')
        
        # dataName = whichData.split("_noiseRatio_")[0].split('ed_')[-1]
        # actualNoise = whichData.split("noiseRatio_")[-1]
        
        # if isTrain is True:
            
        #     dataLoadPath = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/FEed_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}.pkl'
        #     with open(dataLoadPath, "rb") as F:
        #         loadedData = pickle.load(F)
                
        # if isTrain is False:
            
        #     dataLoadPath = f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/FEed_data/{whichData}/test_{dataName}_noised_{actualNoise}.pkl'
        #     with open(dataLoadPath, "rb") as F:
        #         loadedData = pickle.load(F)
                
                
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
    
    elif whichData in FEed_testBed_interval_dataLst:
        
        dataName = whichData.split("_noiseRatio_")[0].split('ed_')[-1]
        actualNoise = whichData.split("noiseRatio_")[-1]
        
        if isTrain is True:
            
            dataLoadPath = f'./800_Testbed/pkled_data/FEed_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}.pkl'
            with open(dataLoadPath, "rb") as F:
                loadedData = pickle.load(F)
                
        if isTrain is False:
            
            dataLoadPath = f'./800_Testbed/pkled_data/FEed_data/{whichData}/test_{dataName}_noised_{actualNoise}.pkl'
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
            
        self.FEed_testBed_interval_dataLst = []
        datatype_lst = ['800_0','800_45']
        interval_lst = [128,256,512,1024,2048,4096]
        noise_ratio_lst = [round(0.1*i,1) for i in range(10)]
        for datatype_1 in datatype_lst:
            for interval_1 in interval_lst:
                for nonoise_ratio in noise_ratio_lst:
                    self.FEed_testBed_interval_dataLst.append(f'FEed_{datatype_1}_interval_{interval_1}_noiseRatio_{nonoise_ratio}')
            
        if self.dataType in self.testBest_intervalLst:
            
            whichData = configs["data_type"]
            dataName = whichData.split("_noiseRatio_")[0]
            actualNoise = whichData.split("noiseRatio_")[-1]
        
            self.mean_std_dict = openPickle(
                f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/noised_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}_mean_std.pkl'
            )
            
        if self.dataType in self.noised_data_lst:
            raise Exception('legacy error ')
        
        elif self.dataType in self.FEed_testBed_interval_dataLst:
            
            whichData = configs["data_type"]
            dataName = whichData.split("_noiseRatio_")[0]
            actualNoise = whichData.split("noiseRatio_")[-1]
            
            self.scaler = load(
                f'./800_Testbed/pkled_data/FEed_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}_scaler.joblib'
            )
            
            
            # whichData = configs["data_type"]
            # dataName = whichData.split("_noiseRatio_")[0]
            # actualNoise = whichData.split("noiseRatio_")[-1]
        
            # self.mean_std_dict = openPickle(
            #     f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/noised_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}_mean_std.pkl'
            # )
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
            
            raise Exception('legacy error 1')
            
            # data = self.loadedData[idx][0]
            # self.doZScore = self.configs['do_zScore']
            # if self.doZScore:
            #     print('tesssssssssssssssssssssssssssssssssssssssssss')
            #     meanValue = self.mean_std_dict['mean']
            #     stdValue = self.mean_std_dict['std']
                
            #     data = (data-meanValue)*(1/(stdValue+1e-9))
            

            # # if self.doFlatten:
            # #     data = torch.flatten(data)

            # label = self.loadedData[idx][1]
            
            # flg = self.loadedData[idx][2]

            # return data, label, flg
        
        elif self.dataType in [f'FEed_800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'FEed_800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]:
            
            raise Exception('legacy error 2')
            
            # data = self.loadedData[idx][0]
            
            # # if self.doZScore:
            # #     print('tesssssssssssssssssssssssssssssssssssssssssss')
            # #     meanValue = self.mean_std_dict['mean']
            # #     stdValue = self.mean_std_dict['std']
                
            # #     data = (data-meanValue)*(1/(stdValue+1e-9))

            # # if self.doFlatten:
            # #     data = torch.flatten(data)

            # label = self.loadedData[idx][1]
            
            # flg = self.loadedData[idx][2]

            # return data, label, flg
            
        
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
        
        elif self.dataType in self.FEed_testBed_interval_dataLst:
            
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
            
            
        elif self.dataType == 'mnist_yes':

            data = self.loadedData[idx][0]

            if self.doFlatten:
                data = torch.flatten(data)

            label = self.loadedData[idx][1]
            # print('do mnist zscore')

            return data, label
        
        elif self.dataType == 'cifar_yes':

            data = self.loadedData[idx][0]

            if self.doFlatten:
                data = torch.flatten(data)
            
            label = self.loadedData[idx][1]
            
            return data, label
        
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
