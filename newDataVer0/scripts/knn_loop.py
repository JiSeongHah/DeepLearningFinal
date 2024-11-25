import os
import numpy as np
import sklearn
from sklearn.svm import OneClassSVM
import pickle
from sklearn.model_selection import train_test_split
from joblib import dump, load 
from data_utils import (
    change_data,
    check_and_normalize,
    dataSetToTensor,
    convert_label_binary,
    return_normal_only,
    dataSetToTensor_testbed
)


class ocsvmLoop:
    def __init__(self, config) -> None:

        self.config = config
        
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

    def runTrain(self, dataSet):
        
        FEed_testBed_dataLst = [f'FEed_800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'FEed_800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]
        
        if self.config['data_type'] in [f'800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]:
            
            raise Exception('legacy error')
            
            # x_train,x_val,y_train,y_val = dataSetToTensor_testbed(dataSet=dataSet,isTrain=True)
            
        elif self.config['data_type'] in FEed_testBed_dataLst:
            
            raise Exception('legacy error')
            
            # x_train,x_val,y_train,y_val = dataSetToTensor_testbed(dataSet=dataSet,isTrain=True)
            
            # whichData = self.config["data_type"]
            # dataName = whichData.split("_noiseRatio_")[0]
            # actualNoise = whichData.split("noiseRatio_")[-1]
            
            # if self.config['do_zScore']:
            #     self.scaler = load(
            #         f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/FEed_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}_scaler.joblib'
            #     )
                
            #     x_train = self.scaler.transform(x_train)
            #     x_val = self.scaler.transform(x_val)
                
        elif self.config['data_type'] in self.FEed_testBed_interval_dataLst:
            
            x_train,x_val,y_train,y_val = dataSetToTensor_testbed(dataSet=dataSet,isTrain=True)
            
            whichData = self.config["data_type"]
            dataName = whichData.split("_noiseRatio_")[0]
            actualNoise = whichData.split("noiseRatio_")[-1]
            
            if self.config['do_zScore']:
                self.scaler = load(
                    f'./800_Testbed/pkled_data/FEed_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}_scaler.joblib'
                )
                
                x_train = self.scaler.transform(x_train)
                x_val = self.scaler.transform(x_val)
            
                
        elif self.config['data_type'] in self.testBest_intervalLst:
            
            x_train,x_val,y_train,y_val = dataSetToTensor_testbed(dataSet=dataSet,isTrain=True)
            
        else:

            x_total, y_total = dataSetToTensor(dataSet=dataSet)

            y_total = convert_label_binary(label_tensor=y_total, config=self.config)

            x_total = check_and_normalize(x_total, self.config)

            x_train, x_val, y_train, y_val = train_test_split(
                x_total, y_total, test_size=0.2, random_state=42
            )

            x_train, y_train = return_normal_only(x_train, y_train)

        # y_val = np.where(y_val == 0 ,1, -1)

        kernel = self.config["ocsvm_kernel"]

        self.ocsvmModel = OneClassSVM(kernel=kernel)

        print(f"ocsvm training start....")
        self.ocsvmModel.fit(X=x_train, y=y_train)
        print(f"ocsvm training complete!!!")

        y_pred = self.ocsvmModel.predict(x_val)

        y_pred = np.where(y_pred == 1, 0, -1)
        y_pred = np.where(y_pred == -1, 1, 0)

        deicisionFunction = self.ocsvmModel.decision_function(X=x_val)

        dec_max = np.max(deicisionFunction)

        y_anomaly_score = dec_max - deicisionFunction

        saveDict = {
            "ocsvm_model": self.ocsvmModel,
            "x_train": x_val,
            "y_train": y_val,
            "y_pred": y_pred,
            "y_anomaly_score": y_anomaly_score,
        }

        print("ssvdd training complete!!!")

        return saveDict

    def runTest(self, dataSet, loadedModel):
        
        FEed_testBed_dataLst = [f'FEed_800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'FEed_800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]
        
        if self.config['data_type'] in [f'800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]:
            
            raise Exception('legacy error')
            
            # x_test,y_test = dataSetToTensor_testbed(dataSet=dataSet,isTrain=False)
            
        elif self.config['data_type'] in FEed_testBed_dataLst:
            
            raise Exception('legacy error')
            
            # x_test,y_test = dataSetToTensor_testbed(dataSet=dataSet,isTrain=False)
            
            # whichData = self.config["data_type"]
            # dataName = whichData.split("_noiseRatio_")[0]
            # actualNoise = whichData.split("noiseRatio_")[-1]
            
            # if self.config['do_zScore']:
            #     self.scaler = load(
            #         f'/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/800_Testbed/pkled_data/FEed_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}_scaler.joblib'
            #     )
                
            #     x_test = self.scaler.transform(x_test)
                
        elif self.config['data_type'] in self.testBest_intervalLst:
            
            x_test,y_test = dataSetToTensor_testbed(dataSet=dataSet,isTrain=False)
            
        elif self.config['data_type'] in self.FEed_testBed_interval_dataLst:
            
            x_test,y_test = dataSetToTensor_testbed(dataSet=dataSet,isTrain=False)
            
            whichData = self.config["data_type"]
            dataName = whichData.split("_noiseRatio_")[0]
            actualNoise = whichData.split("noiseRatio_")[-1]
            
            if self.config['do_zScore']:
                self.scaler = load(
                    f'./800_Testbed/pkled_data/FEed_data/{whichData}/trainVal_{dataName}_noised_{actualNoise}_scaler.joblib'
                )
                
                x_test = self.scaler.transform(x_test)
            
                
        else:

            x_test, y_test = change_data(dataSet=dataSet, config=self.config, mode="all")

            x_test = check_and_normalize(x_test, self.config)

        # y_test = np.where(y_test == 0,1,-1)

        print("ocsvm model loading..")
        self.ocsvmModel = loadedModel["ocsvm_model"]
        print("loading saved ocsvm model complete")
        print("ocsvm test start...")

        y_pred = self.ocsvmModel.predict(X=x_test)
        
        y_pred = np.where(y_pred == 1, 0, -1)
        y_pred = np.where(y_pred == -1, 1, 0)

        deicisionFunction = self.ocsvmModel.decision_function(X=x_test)

        dec_max = np.max(deicisionFunction)

        y_anomaly_score = dec_max - deicisionFunction

        testResult = {
            "y_pred": y_pred,
            "y_anomaly_score": y_anomaly_score,
            "y_test": y_test,
            "x_test": x_test,
            'test_score_raw':y_anomaly_score,
            'test_label_raw':y_test,
        }

        print("ocsvm test complete!!!")
        return testResult

    def saveModel(self, trainResult):

        modelSavePath = self.config["modelSavePath"]

        try:
            os.makedirs(modelSavePath, exist_ok=True)
            print("making save directory complete")
        except:
            raise Exception

        with open(os.path.join(modelSavePath, "ocsvmTrainResult.pkl"), "wb") as f:
            pickle.dump(trainResult, f)

        print(f"saving trained model complete")

    def load_model(self):

        model_load_path = self.config["modelSavePath"]
        try:
            with open(os.path.join(model_load_path, "ocsvmTrainResult.pkl"), "rb") as f:
                loadedDict = pickle.load(f)
        except:
            raise Exception('ocsvm saved model not exist!!!')
        

        return loadedDict
