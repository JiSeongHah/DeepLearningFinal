import import_ipynb
import csv
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam,AdamW
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from scipy.stats import kde
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score, confusion_matrix, precision_recall_curve,precision_score,recall_score,f1_score
from vanillaDSVDD_model import naiveFCN, naivePreAutoEncoder
from torch.utils.data import DataLoader

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
        
        
        for eachEpoch in range(len(self.config['preAE_epoch'])):
            self.trainPreAE(self,model=model,
                           trainDataSet=trainDataSet)
            self.trainPreAEEnd(self)
            
            
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
            
    def trainStep(self,model,trainDataSet):
        
        self.model.to(self.device)
        
        myTrainDataloader=  DataLoader(
            dataset=trainDataSet,
            batch_size=self.config['batch_size']
            shuffle=True,
            drop_last=True
        )
        
        with torch.set_grad_enabled(True):
            
        
        
        
        
        
        
        
        
        
    
    def runTest(self):
        pass
    
    def saveModel(self):
        pass
    
    def loadModel(self):
        pass