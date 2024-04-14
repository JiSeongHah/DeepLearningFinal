import os
import numpy as np
import sklearn
from sklearn.svm import OneClassSVM
import pickle


class ocsvmLoop:
    def __init__(self, config) -> None:

        self.config = config

    def runTrain(self, dataSet):

        x_train = []
        y_train = []

        maxNum = 1000
        flgDict = {}
        for eachData in dataSet:
            if eachData[1] not in flgDict.keys():
                flgDict[eachData[1]] = 1
                x_train.append(eachData[0])
                y_train.append(eachData[1])
            else:
                if flgDict[eachData[1]] >= maxNum:
                    print(f"appending label : {eachData[1]} reached maxNum : {maxNum}")
                    continue
                else:
                    flgDict[eachData[1]] += 1
                    x_train.append(eachData[0])
                    y_train.append(eachData[1])

        x_train = np.stack(x_train)

        x_train_mean = np.mean(x_train)
        x_train_std = np.std(x_train, ddof=1)

        print(f"x_train_mena is : {x_train_mean} while x_train_std is : {x_train_std}")

        x_train = (x_train - x_train_mean) / x_train_std

        y_train = np.stack(y_train).reshape(-1, 1)

        x_train = x_train + 0.1 * np.random.randn(*x_train.shape)

        whichLabelAbnormal = self.config["which_label_abnormal"]
        y_train = np.where(y_train == whichLabelAbnormal, 1, -1)

        kernel = self.config["ocsvm_kernel"]

        self.ocsvmModel = OneClassSVM(kernel=kernel)

        print(f"ocsvm training start....")
        self.ocsvmModel.fit(X=x_train, y=y_train)
        print(f"ocsvm training complete!!!")

        y_pred = self.ocsvmModel.predict(x_train)

        deicisionFunction = self.ocsvmModel.decision_function(X=x_train)

        dec_max = np.max(deicisionFunction)

        y_anomaly_score = dec_max - deicisionFunction

        saveDict = {
            "ocsvm_model": self.ocsvmModel,
            "x_train": x_train,
            "y_train": y_train,
            "y_pred": y_pred,
            "y_anomaly_score": y_anomaly_score,
        }

        print("ssvdd training complete!!!")

        return saveDict

    def runTest(self, dataSet, loadedModel):

        x_test = []
        y_test = []

        maxNum = 100000
        flgDict = {}
        for eachData in dataSet:
            if eachData[1] not in flgDict.keys():
                flgDict[eachData[1]] = 1
                x_test.append(eachData[0])
                y_test.append(eachData[1])
            else:
                if flgDict[eachData[1]] >= maxNum:
                    print(f"appending label : {eachData[1]} reached maxNum : {maxNum}")
                    continue
                else:
                    flgDict[eachData[1]] += 1
                    x_test.append(eachData[0])
                    y_test.append(eachData[1])

        x_test = np.stack(x_test)
        y_test = np.stack(y_test)

        # x_test = x_test + np.random.randint(0,256,(x_test.shape))
        x_test = (x_test - 0.131) / 0.309

        whichLabelAbnormal = self.config["which_label_abnormal"]
        y_test = np.where(y_test == whichLabelAbnormal, 1, -1)

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
