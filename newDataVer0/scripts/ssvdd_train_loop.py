import os
import numpy as np
from SSVDD.ssvdd_train import ssvdd_train
from SSVDD.ssvdd_test import ssvdd_test
import pickle



class sSVDDLoop():
    
    def __init__(self,config) -> None:
        
        self.config= config
        
    def runTrain(self,dataSet):
        
        x_train = []
        y_train = []
        
        for eachData in dataSet:
            x_train.append(eachData[0])
            y_train.append(eachData[1])
            
        x_train = np.stack(x_train)
        y_train = np.stack(y_train)
        
        x_train = x_train + np.random.randint(0,256,(x_train.shape))
        
        whichLabelAbnormal = self.config['whichLabelAbnormal']
        y_train = np.where(y_train==whichLabelAbnormal,1,-1)
        
        iter= self.confg['ssvdd_iter']
        C = self.config['ssvdd_C']
        d = self.config['ssvdd_d']
        eta = self.config['ssvdd_eta']
        kappa = self.config['ssvdd_kappa']
        beta = self.config['ssvdd_beta']
        psi = self.config['ssvdd_psi']
        npt = self.config['ssvdd_npt']
        
        ssvdd_npt, ssvdd_models, ssvdd_Q = ssvdd_train(
            x_train=x_train,
            y_train= y_train,
            iter = iter,
            C= C,
            d = d,
            eta = eta,
            kappa = kappa,
            beta = beta,
            psi = psi,
            npt=  npt
        )
        
        y_pred,y_anomaly_score=  ssvdd_test(
                                            x_train,
                                            y_train,
                                            ssvdd_models[-1],
                                            ssvdd_Q[-1],
                                            ssvdd_npt
        )
        
        
        saveDict= {
            'ssvdd_npt' : ssvdd_npt,
            'ssvdd_models' : ssvdd_models,
            'ssvdd_Q' : ssvdd_Q,
            'x_train' : x_train,
            'y_train' : y_train,
            'y_pred' : y_pred,
            'y_anomaly_score' : y_anomaly_score
        }
        
        print('ssvdd training complete!!!')
        
        return saveDict
        
    def saveModel(self,trainResult):
        
        saveDir = self.config['ssvdd_save_load_path']
        
        try:
            os.makedirs(saveDir)
            print('making save directory complete')
        except:
            raise Exception
        
        with open(os.path.join(saveDir,'ssvddTrainResult.pkl'), 'wb') as f:
            pickle.dump(trainResult, f)
            
        print(f'saving trained model complete')
        
        
    def trainStepEnd(self):
        pass
    
    def load_model(self):
        model_load_path =  self.config['ssvdd_save_load_path']
        
        with open(os.path.join(model_load_path,'ssvddTrainResult.pkl'), 'rb') as f:
            loadedDict = pickle.load(f)
            
        return loadedDict
        
    
    def runTest(self,dataSet,loadedModel):
        
        x_test = []
        y_test = []
        
        for eachData in dataSet:
            x_test.append(eachData[0])
            y_test.append(eachData[1])
            
        x_test = np.stack(x_test)
        y_test = np.stack(y_test)
        
        x_test = x_test + np.random.randint(0,256,(x_test.shape))
        
        whichLabelAbnormal = self.config['whichLabelAbnormal']
        y_test = np.where(y_test==whichLabelAbnormal,1,-1)
        
        
        ssvdd_npt = loadedModel['ssvdd_npt']
        ssvdd_models= loadedModel['ssvdd_models']
        ssvdd_Q = loadedModel['ssvdd_Q']
        
        y_pred,y_anomaly_score = ssvdd_test(
                                            x_test,
                                            y_test,
                                            ssvdd_models[-1],
                                            ssvdd_Q[-1],
                                            ssvdd_npt
        )
        
        testResult = {
            'y_pred':y_pred,
            'y_anomaly_score' :y_anomaly_score,
            'y_test': y_test,
            'x_test' : x_test
        }
        
        return testResult
        
    
    def testStepEnd(self):
        pass
    
# x_train = np.random.randn(512,128)
# print(x_train.shape)
# y_train = np.random.randint(0,2,(512,1))
# print(y_train.shape)
# y_train = np.where(y_train==0,-1,1)
# iter = 30
# C = 0.1
# d = 16
# eta = 0.1
# kappa = 0.8
# beta = 0.01
# psi = 4
# npt = 1
# import os

# print(os.getcwd())


# model = ssvdd_train(x_train = x_train,
#                     y_train = y_train,
#                     iter = iter,
#                     C = C,
#                     d = d,
#                     eta= eta,
#                     kappa= kappa,
#                     beta= beta,
#                     psi = psi,
#                     npt= npt)