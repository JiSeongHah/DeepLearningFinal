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
from sklearn.metrics import roc_auc_score,average_precision_score, confusion_matrix, precision_recall_curve,precision_score,recall_score,f1_score
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
        
        x_train, x_val, y_train, y_val = split_trn_val(self,dataSet)
        
        
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
        
        self.SVDD_preAE.to(self.device)
        self.SVDD_preAE.train()
        
        
        aeTrainDataloader = tqdm(
            DataLoader(aeTrainDataSet,
                        batch_size =self.config['preAE_batch_size'],
                        shuffle=True,
                        drop_last=True
            )
        )
        
        with torch.set_grad_enabled(True):
            
            for idx,bInputDict in enumerate(aeTrainDataloader):
                
                bInput = bInputDict['input']
                
                self.AEoptim.zero_grad()
                                
                answer = bInput.float()
                            
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
        
        self.DSVDD_preAE.to(self.device)
        self.DSVDD_preAE.eval()
        
        z_ = []
        
        tqdm._instances.clear()
        theDloader = tqdm(
            DataLoader(
                normalDataSet,
                batch_size =self.config['preAE_batch_size'],
                shuffle=True,
                drop_last=False
            )
        )
        
        with torch.set_grad_enabled(False):
            for idx, bInput in enumerate(theDloader):
                
                bInput= bInput['input']
            
                self.AEoptim.zero_grad()
                                
                answer = bInput.float()

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
        
        self.DSVDD_preAE.to(self.device)
        self.DSVDD_preAE.eval()
        
        self.DSVDD_model.to(self.device)
        self.DSVDD_model.train()
                
        tqdm._instances.clear()
        theDloader = tqdm(DataLoader(mainModelTrainDataSet,batch_size =self.trnBSizeMain,shuffle=True,num_workers=0),position=0,leave=True)
        
        for idx,bInputDict in enumerate(theDloader):
                
            bInput = bInputDict['input']
            
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
    
        thresholdLst = [i/1000 for i in range(1,1000)]
                
        theDloader = tqdm(
            valDataloader,
            position=0,
            leave=True
        )
        
        with torch.set_grad_enabled(False):
            
            for idx,(totalBInput) in enumerate(theDloader):
                
                bInput = totalBInput['input']
                
                self.Modeloptim.zero_grad()
                                
                bOutput = self.SVDD_model(bInput.float().to(self.device)).cpu()
                
                if self.transformer == 'naive':
                    eachScore = self.calMSELoss(bOutput,self.centre.repeat(bOutput.size(0),1,1),reduction='none')
                    if useMax == True:
                        totalScoreLstVal.append(torch.amax(eachScore,dim=(1,2)))
                    else:
                        totalScoreLstVal.append(torch.mean(eachScore,dim=(1,2)))
                        print(f'totalScoreLstV size : {totalScoreLstVal[-1].size()}')
                else:
                    eachScore = self.calMSELoss(bOutput,self.centre.repeat(bOutput.size(0),1,1),reduction='none')
                    if useMax == True:
                        totalScoreLstVal.append(torch.amax(eachScore,dim=(1,2)))
                    else:
                        totalScoreLstVal.append(torch.mean(eachScore,dim=(1,2)))
                        print(f'totalScoreLstV size : {totalScoreLstVal[-1].size()}')
                
                
                totalLabelLstVal.append(bLabel)
                
                
        totalLabelTrue, totalScores = torch.cat(totalLabelLstVal).numpy(), torch.cat(totalScoreLstVal).numpy()
        
        print(f'shape of label : {totalLabelTrue.shape}')
        print(f'shape of score : {totalScores.shape}')
        
        normalLabelKey = 1 not in totalLabelTrue[self.realNGDataNum+self.visionNGDataNum:]
        anomalyLabelKey = 0 not in totalLabelTrue[:self.realNGDataNum+self.visionNGDataNum]
        lenKey = len(totalLabelTrue) == totalScores.shape[0]
        assert normalLabelKey == True, 'somethings wrong!'
        assert anomalyLabelKey == True, 'somethings wrong!'
        assert lenKey == True, 'somethings wrong!'
        
        print(normalLabelKey,anomalyLabelKey,lenKey)
        print(np.mean(totalLabelTrue[self.realNGDataNum+self.visionNGDataNum:]*1.0))
        print(np.mean(totalLabelTrue[:self.realNGDataNum+self.visionNGDataNum]))
        print(np.sum(totalLabelTrue[self.realNGDataNum+self.visionNGDataNum:]*1.0))
        print(np.sum(totalLabelTrue[:self.realNGDataNum+self.visionNGDataNum]))
        print('every key is okay')
        
        
#         print(f'plotPrecision is : {plotPrecision}')
#         print(f'plotRecall is : {plotRecall}')
#         print(f'plotTHreshold is : {plotThreshold}')
        
#         totalPRSCORE = average_precision_score(totalLabelTrue,totalScores)
#         totalRUCSCORE = roc_auc_score(totalLabelTrue,totalScores)
        
#         print(f'PR Score is : {totalPRSCORE} and RUC score is : {totalRUCSCORE}')
        
        fLst = []
        fOnlyLst = []
        fRealOnlyLst = []
        fRealOnlyOnlyLst = []
        
        
        

        resultPerThresholdLst = []
        resultPerThresholdLstRealOnly = []

        scoreNormal = totalScores[self.realNGDataNum+self.visionNGDataNum:]
        scoreAnomaly = totalScores[:self.realNGDataNum+self.visionNGDataNum]
#         scoreAnomalyRealOnly = totalScores[:self.realNGDataNum]

#         normalLen = int(len(scoreNormal)*eachNormalRatio)

        labelTrueNormal = totalLabelTrue[self.realNGDataNum+self.visionNGDataNum:]
        labelTrueAnomaly = totalLabelTrue[:self.realNGDataNum+self.visionNGDataNum]
#         labelTrueAnomalyRealOnly = totalLabelTrue[:self.realNGDataNum]

        scores = np.concatenate([scoreAnomaly,scoreNormal[:]])
        labelTrue = np.concatenate([labelTrueAnomaly,labelTrueNormal[:]])

        plotPrecision,plotRecall,plotThreshold = precision_recall_curve(labelTrue,scores)
        fpr,tpr,rocThreshold = roc_curve(labelTrue,scores)

#         scoresRealOnly = np.concatenate([scoreAnomalyRealOnly,scoreNormal[:normalLen]])
#         labelTrueRealOnly = np.concatenate([labelTrueAnomalyRealOnly,labelTrueNormal[:normalLen]])

        print(f'min Score is : {min(scores)} while max Score is : {max(scores)}')
        saveMin = min(scores)
        saveMax = max(scores)
        scores = (scores-min(scores)) / (max(scores)-min(scores))

#             print(f'min Score is : {min(scores)} while max Score is : {max(scores)}')
#         scoresRealOnly = np.nan_to_num((scoresRealOnly-min(scoresRealOnly)) / (max(scoresRealOnly)-min(scoresRealOnly)+1e-8))

        totalPRSCORE = average_precision_score(labelTrue,scores)
#         realonlyPRESCORE = average_precision_score(labelTrueRealOnly,scoresRealOnly)

        totalRUCSCORE = roc_auc_score(labelTrue,scores)
#         realOnlyRUCSCORE = roc_auc_score(labelTrueRealOnly,scoresRealOnly)


#             scores = np.nan_to_num(scores)


        for eachThreshold in thresholdLst:

            labelPred = np.where(scores >= eachThreshold, 1,0)

            tn, fp, fn, tp = confusion_matrix(labelTrue,labelPred).ravel()

            precisionScore = precision_score(labelTrue,labelPred)
            recallScore = recall_score(labelTrue,labelPred)
            f1Score = f1_score(labelTrue,labelPred)
            fOnlyLst.append(f1Score)
            resultPerThresholdLst.append([eachThreshold,tn,fp,fn,tp,precisionScore,recallScore,f1Score])

#                 labelPredRealOnly = np.where(scoresRealOnly >= eachThreshold, 1,0)

#                 tn, fp, fn, tp = confusion_matrix(labelTrueRealOnly,labelPredRealOnly).ravel()

#                 precisionScore = precision_score(labelTrueRealOnly,labelPredRealOnly)
#                 recallScore = recall_score(labelTrueRealOnly,labelPredRealOnly)
#                 f1Score = f1_score(labelTrueRealOnly,labelPredRealOnly)

#                 resultPerThresholdLstRealOnly.append([eachNormalRatio,eachThreshold,tn,fp,fn,tp,precisionScore,recallScore,f1Score])


#                 fRealOnlyOnlyLst.append(f1Score)




            if printAll ==True:
                print(resultPerThresholdLst[-1],'total')

#             for i in resultPerThresholdLstRealOnly:
#                 print(i,'realOnly')


        fMax = resultPerThresholdLst[fOnlyLst.index(max(fOnlyLst))]
#             fMaxRealOnly = resultPerThresholdLstRealOnly[fRealOnlyOnlyLst.index(max(fRealOnlyOnlyLst))]



        if doSave == True:

            with open(self.plotSaveDir+f'totalTestsvddResultPerRatio{eachNormalRatio}_Threshold_{self.task}_{self.scaleMethod}.csv','w',newline='') as F:
                wr = csv.writer(F)
                wr.writerow(['eachThreshold','tn','fp','fn','tp','precisionScore','recallScore','f1Score'])
                wr.writerows(resultPerThresholdLst)



        print('mission complete')

        self.labelLstVal.clear()
        self.scoreLstVal.clear()

    
        self.SVDD_model.to('cpu')
        
        print([totalPRSCORE , totalRUCSCORE ,fMax[-1],fMax])
        
        return [totalPRSCORE , totalRUCSCORE,fMax[-1] ,fMax,plotPrecision,plotRecall,plotThreshold,fpr,tpr,rocThreshold,saveMin,saveMax]
        
        
        
    def validationStep(self,useMax=False,doSave=False,printAll=True):
                
        
            
    
    def runTest(self):
        pass
    
    def saveModel(self):
        pass
    
    def loadModel(self):
        pass