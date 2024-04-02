import numpy as np
from ssvdd_train_loop import sSVDDLoop
from mySSVDD.ssvdd_train import ssvdd_train
from mySSVDD.ssvdd_test import ssvdd_test
from Dataload import myNewDataset
from sklearn.metrics import confusion_matrix ,roc_auc_score, average_precision_score, f1_score,precision_score,recall_score
from datetime import datetime



class MainLoop():
    
    def __init__(self,config,doTestOnly) -> None:
        
        self.config = config
        
        self.doTestOnly =doTestOnly
    
    def runMainLoop(self):
        
        if self.doTestOnly == False:
        
            trainResult = self.runTrainLoop()
            
            self.trainResultToConfig(trainResult=trainResult)
        
        testResult = self.runTestLoop()
        
        self.testResultToConfig(testResult=testResult)
        
        
        return self.config
        

    def runTrainLoop(self):
        
        
        trainDataSet = myNewDataset(
            configs = self.config,
            isTrain= True
        )
        
        whichModel = self.config['which_model']
        
        if whichModel == 'ssvdd':
            
            model = sSVDDLoop(config=self.config)
            
        trainResult = model.runTrain(dataSet=trainDataSet)
        
        model.saveModel(trainResult=trainResult)
            
        print('training complete')
        
        return trainResult
    
    def trainResultToConfig(self,trainResult):
        
        y_pred= trainResult['y_pred']
        y_anomaly_score = trainResult['y_anomaly_score']
        y_train = trainResult['y_train']
        
        y_pred = np.where(y_pred==-1,0,1)
        y_train = np.where(y_train==-1,0,1)
        
        tn, fp, fn, tp = confusion_matrix(y_pred=y_pred,
                                           y_true=y_train).ravel()
        
        
        
        precisionScore= precision_score(y_pred=y_pred,
                                        y_true=y_train)
        
        recallScore = recall_score(y_pred=y_pred,
                                    y_true =y_train)
        
        f1Score=  f1_score(y_pred=y_pred,
                           y_true=y_train)
        
        averagePrecision = average_precision_score(y_score=y_anomaly_score,
                                                   y_true=y_train)
        
        rocAuc = roc_auc_score(y_score=y_anomaly_score,
                               y_true=y_train)
        
        self.config['trainResult'] = {}
        
        self.config['trainResult']['truePos'] = str(tp)
        self.config['trainResult']['trueNeg'] = str(tn)
        self.config['trainResult']['falsePos'] = str(fp)
        self.config['trainResult']['falseNeg'] = str(fn)
        
        self.config['trainResult']['precision'] = str(precisionScore)
        self.config['trainResult']['recall'] = str(recallScore)
        self.config['trainResult']['f1'] = str(f1Score)
        
        self.config['trainResult']['averagePrecision'] = str(averagePrecision)
        self.config['trainResult']['rocAuc'] = str(rocAuc)
        
        self.config['trainResult']['time'] = datetime.now()
        
        
    def runTestLoop(self):
        
        
        testDataSet = myNewDataset(
            configs = self.config,
            isTrain= False
        )
        
        whichModel = self.config['which_model']
        
        if whichModel == 'ssvdd':
            
            model = sSVDDLoop(config=self.config)
            
        loadedModel = model.load_model()
        
        testResult = model.runTest(dataSet=testDataSet,
                            loadedModel=loadedModel)
            
        print('test complete')
        
        return testResult
    
    def testResultToConfig(self,testResult):
        
        y_pred= testResult['y_pred']
        y_anomaly_score = testResult['y_anomaly_score']
        y_test = testResult['y_test']
        
        y_pred = np.where(y_pred==-1,0,1)
        y_test = np.where(y_test==-1,0,1)
        
        tn, fp, fn, tp = confusion_matrix(y_pred=y_pred,
                                           y_true=y_test).ravel()
        
        
        
        precisionScore= precision_score(y_pred=y_pred,
                                        y_true=y_test)
        
        recallScore = recall_score(y_pred=y_pred,
                                    y_true =y_test)
        
        f1Score=  f1_score(y_pred=y_pred,
                           y_true=y_test)
        
        averagePrecision = average_precision_score(y_score=y_anomaly_score,
                                                   y_true=y_test)
        
        rocAuc = roc_auc_score(y_score=y_anomaly_score,
                               y_true=y_test)
        
        self.config['testResult'] = {}
        
        self.config['testResult']['truePos'] = str(tp)
        self.config['testResult']['trueNeg'] = str(tn)
        self.config['testResult']['falsePos'] = str(fp)
        self.config['testResult']['falseNeg'] = str(fn)
        
        self.config['testResult']['precision'] = str(precisionScore)
        self.config['testResult']['recall'] = str(recallScore)
        self.config['testResult']['f1'] = str(f1Score)
        
        self.config['testResult']['averagePrecision'] = str(averagePrecision)
        self.config['testResult']['rocAuc'] = str(rocAuc)
        
        self.config['testResult']['time'] = datetime.now()
        
        
            
            
            
            
        
        
        
        
        
        



