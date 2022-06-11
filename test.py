from torch.optim import AdamW, Adam,SGD
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset,TensorDataset
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.svm import SVC
import pickle
from torch.nn import Parameter
import math
from sklearn.neighbors import KNeighborsClassifier


def DNNVer(modelLoadDir,
           dataLoadDir,
           testResultSaveDir,
           dataScale,
           gpuUse=True):

    predictor = torch.load(modelLoadDir)

    ############ test Data Config ###################################
    testData = np.loadtxt(dataLoadDir, delimiter=',')
    if dataScale == True:
        scaler = preprocessing.StandardScaler().fit(testData)
        testData = scaler.transform(testData)

    testDataset = TensorDataset(torch.from_numpy(testData))
    tstDataloader = DataLoader(testDataset, batch_size=1, shuffle=False)
    ############ test Data Config ###################################

    ##### use GPU or not ###########################
    if gpuUse == True:
        USE_CUDA = torch.cuda.is_available()
        print(USE_CUDA)
        device = torch.device('cuda' if USE_CUDA else 'cpu')
        print('학습을 진행하는 기기:', device)
    else:
        device = torch.device('cpu')
        print('학습을 진행하는 기기:', device)
    ##### use GPU or not ###########################

    predictor.model.to(device)

    predictor.model.eval()
    TestDataloader = tqdm(tstDataloader)

    results = []

    with torch.set_grad_enabled(False):

        for idx,(inputs,) in enumerate(TestDataloader):

            inputs = inputs.float()
            inputs = inputs.to(device)
            logits = predictor.forward(inputs).cpu()

            preds = torch.argmax(logits,dim=1)

            results.append(str(preds.item()))
            TestDataloader.set_description(f'Processing : {idx} of {len(TestDataloader)} with pred : {preds.item()}')

    with open(testResultSaveDir+'test_output_하지성.txt','w') as FF:
        for eachResult in results:
            FF.write(eachResult+'\n')



def SVCVer(modelLoadDir,
           dataLoadDir,
           testResultSaveDir,
           dataScale):

    with open(modelLoadDir,'rb') as ff:
        svcClf = pickle.load(ff)

    ############ test Data Config ###################################
    testData = np.loadtxt(dataLoadDir, delimiter=',')
    if dataScale == True:
        scaler = preprocessing.StandardScaler().fit(testData)
        testData = scaler.transform(testData)

    ############ test Data Config ###################################

    predictedLabel = svcClf.predict(testData)
    results = []

    for each in predictedLabel:
        results.append(str(each))

    with open(testResultSaveDir+'test_output_하지성.txt','w') as FF:
        for eachResult in results:
            FF.write(eachResult+'\n')

def arcFaceVer(modelLoadDir,
               dataLoadDir,
               testResultSaveDir,
               dataScale,
               gpuUse=True):

    predictor = torch.load(modelLoadDir)

    ############ test Data Config ###################################
    testData = np.loadtxt(dataLoadDir, delimiter=',')
    if dataScale == True:
        scaler = preprocessing.StandardScaler().fit(testData)
        testData = scaler.transform(testData)

    testDataset = TensorDataset(torch.from_numpy(testData))
    tstDataloader = DataLoader(testDataset, batch_size=1, shuffle=False)
    ############ test Data Config ###################################

    ##### use GPU or not ###########################
    if gpuUse == True:
        USE_CUDA = torch.cuda.is_available()
        print(USE_CUDA)
        device = torch.device('cuda' if USE_CUDA else 'cpu')
        print('학습을 진행하는 기기:', device)
    else:
        device = torch.device('cpu')
        print('학습을 진행하는 기기:', device)
    ##### use GPU or not ###########################

    predictor.model.to(device)

    predictor.model.eval()
    TestDataloader = tqdm(tstDataloader)
    TDataloader = tqdm(predictor.trnDataloader)

    results = []

    criterionFeatures = []
    criterionLabels = []

    with torch.set_grad_enabled(False):
        for idx, (bInput, bLabel) in enumerate(TDataloader):
            bLabel = bLabel.long()
            bInput = bInput.float()
            bInput = bInput.to(device)

            bFeature = predictor.forward(bInput).cpu()

            criterionFeatures.append(bFeature.clone().detach())
            criterionLabels.append(bLabel)

    criterionFeatures = torch.cat(criterionFeatures).numpy()
    criterionLabels = torch.cat(criterionLabels).numpy()

    knnClf = KNeighborsClassifier(n_neighbors=predictor.topkNum)

    compareFeatures = []
    compareGTLabels = []

    with torch.set_grad_enabled(False):
        for idx, (valBInput, valBLabel) in enumerate(TestDataloader):
            valBInput = valBInput.float()
            valBInput = valBInput.to(device)
            valBFeature = predictor.forward(valBInput).cpu()

            compareFeatures.append(valBFeature.clone().detach())
            compareGTLabels.append(valBLabel)

    compareFeatures = torch.cat(compareFeatures).numpy()
    compareGTLabels = torch.cat(compareGTLabels).numpy()

    print(criterionFeatures.shape, criterionLabels.shape)
    print(compareFeatures.shape, compareGTLabels.shape)
    # print(compareFeatures,compareGTLabels)

    knnClf.fit(criterionFeatures, criterionLabels)
    predictedLabels = knnClf.predict(compareFeatures)

    for each in predictedLabels:
        results.append(str(each))

    with open(testResultSaveDir+'test_output_하지성.txt','w') as FF:
        for eachResult in results:
            FF.write(eachResult+'\n')







