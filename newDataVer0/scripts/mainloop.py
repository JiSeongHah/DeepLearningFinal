import numpy as np
from SSVDD.ssvdd_train import ssvdd_train
from SSVDD.ssvdd_test import ssvdd_test
from Dataload import myNewDataset
from sklearn.metrics import confusion_matrix ,roc_auc_score, average_precision_score, f1_score,precision_score,recall_score
from datetime import datetime



class MainLoop():
    
    def __init__(self,config,doTestOnly) -> None:
        
        self.config = config
        
        self.doTestOnly =doTestOnly
    
    def runMainLoop(self):
        
        if self.doTestOnly == False:
        
            self.runTrainLoop()
        
        testResult = self.runTestLoop()
        
        y_pred= testResult['y_pred']
        y_anomaly_score = testResult['y_anomaly_score']
        y_test = testResult['y_test']
        
        y_test = np.where(y_test==-1,0,1)
        y_pred = np.where(y_pred==-1,0,1)
        
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
        
        self.config['testResult']['truePos'] = tp
        self.config['testResult']['trueNeg'] = tn
        self.config['testResult']['falsePos'] = fp
        self.config['testResult']['falseNeg'] = fn
        
        self.config['testResult']['precision'] = precisionScore
        self.config['testResult']['recall'] = recallScore
        self.config['testResult']['f1'] = f1Score
        
        self.config['testResult']['averagePrecision'] = averagePrecision
        self.config['testResult']['rocAuc'] = rocAuc
        
        self.config['time'] = datetime.now()
        
        return self.config
        

    def runTrainLoop(self):
        
        whichData = self.config['whichData']
        
        trainDataSet = myNewDataset(
            configs = self.config,
            isTrain= True
        )
        
        whichModel = self.config['whichModel']
        
        if whichModel == 'ssvdd':
            
            model = sSVDDLoop(config=self.config)
            
        trainResult = runTrain(dataSet=trainDataSet)
        
        model.saveModel(trainResult=trainResult)
            
        print('training complete')
        
        return trainResult
        
    def runTestLoop(self):
        
        whichData = self.config['whichData']
        
        testDataSet = myNewDataset(
            configs = self.config,
            isTrain= False
        )
        
        whichModel = self.config['whichModel']
        
        if whichModel == 'ssvdd':
            
            model = sSVDDLoop(config=self.config)
            
        loadedModel = model.load_model()
        
        testResult = runTest(dataSet=testDataSet,
                            loadedModel=loadedModel)
            
        print('test complete')
        
        return testResult
        
        
            
            
            
            
        
        
        
        
        
        



