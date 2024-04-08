import import_ipynb
import csv
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam,AdamW
from torch.utils.data import Dataset,DataLoader,TensorDataset
from tqdm import tqdm
from scipy.stats import kde
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score, confusion_matrix,precision_score,recall_score,f1_score
from vanillaDSVDD_model import naiveFCN, naivePreAutoEncoder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class vanillaDsvddLoop():
    def __init__(self,config) -> None:
        self.config = config
        
        self.trainLossTmp = []
        self.trainAccTmp = []
        
        self.trainLoss = []
        self.trainAcc = []
        
        self.valLossTmp = []
        self.valAccTmp = []
        
        self.valLoss = []
        self.valAcc = []
        
        self.testLossTmp = []
        self.testAccTmp = []
        
        self.testLoss = []
        self.testAcc = []
        
        self.AElossLstTrnTmp= []
        self.AElossLstTrn= []
        
        self.modelLossLstTrnTmp = []
        self.modelLossLstTrn = []
        
        
    
    def runTrain(self,dataSet):
        
        USE_CUDA = torch.cuda.is_available()
        print(USE_CUDA)

        self.device = torch.device('cuda:0' if USE_CUDA else 'cpu')
        print('학습을 진행하는 기기:',self.device)
        
        self.DSVDD_model = naiveFCN(
            hDim1=self.config['hDim1'],
            hDim2=self.config['hDim2'],
            hDim3=self.config['hDim3'],
            FVSize=self.config['FVSize'],
            inputSize=self.config['inputSize']
        )
        
        self.DSVDD_preAE = naivePreAutoEncoder(
            hDim1=self.config['hDim1'],
            hDim2=self.config['hDim2'],
            hDim3=self.config['hDim3'],
            FVSize=self.config['FVSize'],
            inputSize=self.config['inputSize']
        )
        
        self.Modeloptim = AdamW(self.DSVDD_model.parameters(),
                              lr =3e-4,
                              weight_decay=0.5e-6)
        
        self.AEoptim = AdamW(self.DSVDD_preAE.parameters(),
                            lr =3e-4,
                            weight_decay =0.5e-3)
        
        x_train, x_val, y_train, y_val = self.split_trn_val(dataSet)
        
        
        for eachEpoch in range(len(self.config['preAE_epoch'])):
            self.trainPreAE(aeTrainDataSet=(x_train,y_train))
            self.trainPreAEEnd()
        
        self.saveWeightAE(iterNum=len(self.config['preAE_epoch']))
        self.transferAEtoMainModel()
        
        centre= self.setCentre(self,normalDataSet=(x_train,y_train))
        
        for eachEpoch in range(len(self.config['mainModel_epoch'])):
            self.trainMainModel(aeTrainDataSet=(x_train,y_train))
            self.trainModelEnd()
            self.validationStep(validationDataset= (x_val,y_val))
            self.validationStepEnd()
            
    def split_trn_val(self,dataSet):
        
        x_train_total = []
        y_train_total = []
        
        for eachData in dataSet:
            x_train_total.append(eachData[0])
            y_train_total.append(eachData[1])
            

        x_train_total = np.stack(x_train_total)
        y_train_total = np.stack(y_train_total)
        
        whichLabelAbnormal = self.config['which_label_abnormal']
        y_train_total = np.where(y_train_total==whichLabelAbnormal,1,0)
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_total, y_train_total, test_size=0.1, random_state=42
        )
        
        return x_train, x_val, y_train, y_val
            
    def calMSELoss(self,output,label,reduction='mean'):
        
        loss = nn.MSELoss(reduction=reduction)
        
        return loss(output,label)
    
    def trainPreAE(self,aeTrainDataSet):
        
        x_train, y_train = aeTrainDataSet[0], aeTrainDataSet[1]
        
        aeTrainTensorDataSet = TensorDataset(x_train,y_train)
        
        self.DSVDD_preAE.to(self.device)
        self.DSVDD_preAE.train()
        
        aeTrainDataloader = tqdm(
            DataLoader(aeTrainTensorDataSet,
                        batch_size =self.config['preAE_batch_size'],
                        shuffle=True,
                        drop_last=True
            )
        )
        
        with torch.set_grad_enabled(True):
            
            for idx,bInputLabel in enumerate(aeTrainDataloader):
                
                bInput = bInputLabel[0]
                
                self.AEoptim.zero_grad()
                                
                answer = bInput.clone().detach().float()
                            
                bOutput = self.SVDD_preAE(bInput.float().to(self.device)).cpu()
                
                loss = self.calMSELoss(bOutput,answer)
                
                loss.backward()
                self.AEoptim.step()
                
                self.AElossLstTrnTmp.append(loss.item())
                  
        self.DSVDD_preAE.to('cpu')
        self.DSVDD_preAE.eval()
        
    def trainPreAEEnd(self):
        
        self.AElossLstTrn.append(np.mean(self.AElossLstTrnTmp))
        
        plt.plot(range(len(self.AElossLstTrn)),self.AElossLstTrn)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Auto Encoder Train Loss')
        plt.savefig(os.path.join(self.config['plotSaveDir'],'aeTrainLossPlot.png'),dpi=200)
        plt.cla()
        plt.clf()
        plt.close()
        
        self.AElossLstTrnTmp.clear()
        
    def saveWeightAE(self,iterNum):
        
        torch.save(self.DSVDD_preAE.state_dict(), os.path.join(self.config['ae_save_load_path'],'ae_',str(iterNum))+'.pt')
        
        print('saving AE weight complete')
        
        
    def saveWeightMainModel(self,iterNum):
        
        torch.save(self.DSVDD_model.state_dict(),os.path.join(self.config['mainmodel_save_load_path'],'ae_',str(iterNum))+'.pt')
        
        
        print('saving MainModel weight complete')
        
    def transferAEtoMainModel(self):
        
        print('transferring weight of auto encoder to main model')
        
        self.DSVDD_model.load_state_dict(self.DSVDD_preAE.state_dict(),strict=False)
        print('transferring weight of auto encoder to main model complete !!!')
        
    def setCentre(self,normalDataSet):
        
        x_train, y_train = normalDataSet[0], normalDataSet[1]
        
        aeTrainTensorDataSet = TensorDataset(x_train,y_train)
        
        self.DSVDD_preAE.to(self.device)
        self.DSVDD_preAE.eval()
        
        z_ = []
        
        tqdm._instances.clear()
        aeTrainDataloader = tqdm(
            DataLoader(aeTrainTensorDataSet,
                        batch_size =self.config['preAE_batch_size'],
                        shuffle=True,
                        drop_last=True
            )
        )
        
        with torch.set_grad_enabled(False):
            for idx, bInputLabel in enumerate(aeTrainDataloader):
                
                bInput= bInputLabel[0]
            
                self.AEoptim.zero_grad()
                                
                answer = bInput.clone().detach().float()

                eachZ = self.DSVDD_preAE.doEncode(bInput.float().to(self.device)).cpu()
                
                z_.append(eachZ.clone().detach())
                
        z_ = torch.cat(z_)
        
        c = torch.mean(z_ , dim=0)
             
        self.DSVDD_preAE.to('cpu')
        
        cSave = c.numpy()
        np.save(os.path.join(self.config['mainmodel_save_load_path'],'cSave.npy'),cSave)
        tqdm._instances.clear()
        
        self.centre = c
        
        self.DSVDD_preAE.to('cpu')
        return c
    
    def trainMainModel(self,mainModelTrainDataSet):
        
        x_train, y_train = mainModelTrainDataSet[0], mainModelTrainDataSet[1]
        
        mainModelTrainTensorDataSet = TensorDataset(x_train, y_train)
        
        self.DSVDD_preAE.to(self.device)
        self.DSVDD_preAE.eval()
        
        self.DSVDD_model.to(self.device)
        self.DSVDD_model.train()
                
        tqdm._instances.clear()
        theDloader = tqdm(
            DataLoader(mainModelTrainTensorDataSet,batch_size =self.trnBSizeMain,shuffle=True,num_workers=0),
            position=0,
            leave=True
        )
        
        for idx,bInputLabel in enumerate(theDloader):
                
            bInput = bInputLabel[0]
            
            self.Modeloptim.zero_grad()

            with torch.set_grad_enabled(True):
                bOutput = self.SVDD_model(bInput.float().to(self.device)).cpu()

            loss = self.calMSELoss(bOutput,self.centre.repeat(bOutput.size(0),1))

            loss.backward()
            self.Modeloptim.step()

            self.modelLossLstTrnTmp.append(loss.item())
        
        self.DSVDD_model.to('cpu')
        self.DSVDD_model.eval()
        
    def trainModelEnd(self):
        
        self.modelLossLstTrn.append(np.mean(self.modelLossLstTrnTmp))
        
        self.modelLossLstTrnTmp.clear()
        
    def validationStep(self,validationDataSet):
        
        x_val,y_val = validationDataSet[0], validationDataSet[1]
        
        valTensorDataSet = TensorDataset(x_val,y_val)
        
        valDataloader= DataLoader(
            valTensorDataSet,
            batch_size=self.config['preAE_batch_size'],
            shuffle=False,
            drop_last=False
        )
        
    
        totalScoreLstVal = []
        totalLabelLstVal  =[]
         
        self.DSVDD_model.to(self.device)
        self.DSVDD_model.eval()
    
        
        theDloader = tqdm(
            valDataloader,
            position=0,
            leave=True
        )
        
        with torch.set_grad_enabled(False):
            
            for idx,(totalBInput) in enumerate(theDloader):
                
                bInput, bLabel = totalBInput[0], totalBInput[1]
                
                self.Modeloptim.zero_grad()
                                
                bOutput = self.DSVDD_model(bInput.float().to(self.device)).cpu()
                
                eachScore = self.calMSELoss(bOutput,self.centre.repeat(bOutput.size(0),1),reduction='none')
                
                useMax=  self.config['useMax_when_val']
                if useMax == True:
                    totalScoreLstVal.append(torch.amax(eachScore,dim=(1,2)))
                else:
                    totalScoreLstVal.append(torch.mean(eachScore,dim=(1,2)))
                    
                totalLabelLstVal.append(bLabel)
                              
        totalLabelTrue, totalScores = torch.cat(totalLabelLstVal).numpy(), torch.cat(totalScoreLstVal).numpy()
        
        print(f'shape of label : {totalLabelTrue.shape}')
        print(f'shape of score : {totalScores.shape}')
        
        
        print(f'min Score is : {min(totalScores)} while max Score is : {max(totalScores)}')
        saveMin = min(totalScores)
        saveMax = max(totalScores)
        minMaxedScore = (totalScores-min(totalScores)) / (max(totalScores)-min(totalScores))

        averagePrecisionScore = average_precision_score(totalLabelTrue,minMaxedScore)
        rocAucScore = roc_auc_score(totalLabelTrue,minMaxedScore)
        
        fLst = []
        resultPerThresholdLst = []

        thresholdLst = [i/1000 for i in range(1,1000)]
        for eachThreshold in thresholdLst:

            labelPred = np.where(totalScores >= eachThreshold, 1,0)

            tn, fp, fn, tp = confusion_matrix(totalLabelTrue,labelPred).ravel()

            precisionScore = precision_score(totalLabelTrue,labelPred)
            recallScore = recall_score(totalLabelTrue,labelPred)
            f1Score = f1_score(totalLabelTrue,labelPred)
            fLst.append(f1Score)
            resultPerThresholdLst.append([eachThreshold,tn,fp,fn,tp,precisionScore,recallScore,f1Score])

            if printAll ==True:
                print(resultPerThresholdLst[-1])

        fMax = resultPerThresholdLst[fLst.index(max(fLst))]

        print('mission complete')

        self.DSVDD_model.to('cpu')
        
        print([averagePrecisionScore , rocAucScore ,fMax[-1],fMax])
        
        return [averagePrecisionScore,rocAucScore,fMax,resultPerThresholdLst]
        
    def runTest(self):
        
        USE_CUDA = torch.cuda.is_available()
        print(USE_CUDA)

        self.device = torch.device('cuda:0' if USE_CUDA else 'cpu')
        print('학습을 진행하는 기기:',self.device)
        
        self.DSVDD_model = naiveFCN(
            hDim1=self.config['hDim1'],
            hDim2=self.config['hDim2'],
            hDim3=self.config['hDim3'],
            FVSize=self.config['FVSize'],
            inputSize=self.config['inputSize']
        )
        
        self.DSVDD_preAE = naivePreAutoEncoder(
            hDim1=self.config['hDim1'],
            hDim2=self.config['hDim2'],
            hDim3=self.config['hDim3'],
            FVSize=self.config['FVSize'],
            inputSize=self.config['inputSize']
        )
        
        
        x_train, x_val, y_train, y_val = self.split_trn_val(dataSet)
        
        
        for eachEpoch in range(len(self.config['preAE_epoch'])):
            self.trainPreAE(aeTrainDataSet=(x_train,y_train))
            self.trainPreAEEnd()
        
        self.saveWeightAE(iterNum=len(self.config['preAE_epoch']))
        self.transferAEtoMainModel()
        
        centre= self.setCentre(self,normalDataSet=(x_train,y_train))
        
        for eachEpoch in range(len(self.config['mainModel_epoch'])):
            self.trainMainModel(aeTrainDataSet=(x_train,y_train))
            self.trainModelEnd()
            self.validationStep(validationDataset= (x_val,y_val))
            self.validationStepEnd()
        
    
    def saveModel(self):
        pass
    
    def loadModel(self):
        pass