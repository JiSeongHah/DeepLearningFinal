import os
import numpy as np
import sklearn
from sklearn.svm import OneClassSVM
import pickle
from sklearn.model_selection import train_test_split

from data_utils import change_data,check_and_normalize,dataSetToTensor,convert_label_binary,return_normal_only
class ocsvmLoop:
    def __init__(self, config) -> None:

        self.config = config

    def runTrain(self, dataSet):

        x_total, y_total = dataSetToTensor(dataSet=dataSet)
        
        y_total = convert_label_binary(label_tensor=y_total,config=self.config)

        x_total = check_and_normalize(x_total,self.config)
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_total, y_total, test_size=0.2, random_state=42
        )
        
        x_train, y_train = return_normal_only(x_train,y_train)
        
        y_val = np.where(y_val ==0 ,1, -1)

        kernel = self.config["ocsvm_kernel"]

        self.ocsvmModel = OneClassSVM(kernel=kernel)

        print(f"ocsvm training start....")
        self.ocsvmModel.fit(X=x_train, y=y_train)
        print(f"ocsvm training complete!!!")

        y_pred = self.ocsvmModel.predict(x_val)

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

        x_test, y_test = change_data(dataSet=dataSet,config=self.config,mode='all')
        
        x_test = check_and_normalize(x_test,self.config)
        
        y_test = np.where(y_test == 0,1,-1)

        print("ocsvm model loading..")
        self.ocsvmModel = loadedModel["ocsvm_model"]
        print("loading saved ocsvm model complete")
        print("ocsvm test start...")
        
        y_pred = self.ocsvmModel.predict(X=x_test)

        deicisionFunction = self.ocsvmModel.decision_function(X=x_test)

        dec_max = np.max(deicisionFunction)

        y_anomaly_score = dec_max - deicisionFunction

        testResult = {
            "y_pred": y_pred,
            "y_anomaly_score": y_anomaly_score,
            "y_test": y_test,
            "x_test": x_test,
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

        with open(os.path.join(model_load_path, "ocsvmTrainResult.pkl"), "rb") as f:
            loadedDict = pickle.load(f)

        return loadedDict
