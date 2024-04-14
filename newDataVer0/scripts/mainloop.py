import numpy as np
from ssvdd_loop import sSVDDLoop
from ocsvm_loop import ocsvmLoop
from vanillaDSVDD_loop import vanillaDsvddLoop
from mySSVDD.ssvdd_train import ssvdd_train
from mySSVDD.ssvdd_test import ssvdd_test
from smoothedDSVDD_loop import smoothedDsvddLoop
from denoisingDSVDD_loop import denoisingDsvddLoop


from Dataload import myNewDataset

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from datetime import datetime


class MainLoop:

    def __init__(self, config, doTestOnly) -> None:

        self.config = config

        self.doTestOnly = doTestOnly

    def runMainLoop(self):

        if self.doTestOnly == False:

            trainResult = self.runTrainLoop()

            self.trainResultToConfig(trainResult=trainResult)

        testResult = self.runTestLoop()

        self.testResultToConfig(testResult=testResult)

        return self.config

    def runTrainLoop(self):

        trainDataSet = myNewDataset(configs=self.config, isTrain=True)

        whichModel = self.config["which_model"]

        if whichModel == "ssvdd":

            model = sSVDDLoop(config=self.config)

        elif whichModel == "ocsvm":
            model = ocsvmLoop(config=self.config)

        elif whichModel == "vanilla_dsvdd":
            model = vanillaDsvddLoop(config=self.config)
            
        elif whichModel == 'smoothed_dsvdd':
            model = smoothedDsvddLoop(config=self.config)
        
        elif whichModel == 'denoising_dsvdd':
            model = denoisingDsvddLoop(config=self.config)

        trainResult = model.runTrain(dataSet=trainDataSet)

        model.saveModel(trainResult=trainResult)

        print("training complete")

        return trainResult

    def trainResultToConfig(self, trainResult):
        
        whichModel = self.config["which_model"]
        
        assert whichModel in ["ssvdd", "ocsvm","vanilla_dsvdd","smoothed_dsvdd",'denoising_dsvdd']

        if whichModel in ["ssvdd", "ocsvm"]:

            y_pred = trainResult["y_pred"]
            y_anomaly_score = trainResult["y_anomaly_score"]
            y_train = trainResult["y_train"]

            y_pred = np.where(y_pred == -1, 0, 1)
            y_train = np.where(y_train == -1, 0, 1)

            tn, fp, fn, tp = confusion_matrix(y_pred=y_pred, y_true=y_train).ravel()

            precisionScore = precision_score(y_pred=y_pred, y_true=y_train)

            recallScore = recall_score(y_pred=y_pred, y_true=y_train)

            f1Score = f1_score(y_pred=y_pred, y_true=y_train)

            averagePrecision = average_precision_score(
                y_score=y_anomaly_score, y_true=y_train
            )

            rocAuc = roc_auc_score(y_score=y_anomaly_score, y_true=y_train)

            self.config["trainResult"] = {}

            self.config["trainResult"]["truePos"] = str(tp)
            self.config["trainResult"]["trueNeg"] = str(tn)
            self.config["trainResult"]["falsePos"] = str(fp)
            self.config["trainResult"]["falseNeg"] = str(fn)

            self.config["trainResult"]["precision"] = str(precisionScore)
            self.config["trainResult"]["recall"] = str(recallScore)
            self.config["trainResult"]["f1"] = str(f1Score)

            self.config["trainResult"]["averagePrecision"] = str(averagePrecision)
            self.config["trainResult"]["rocAuc"] = str(rocAuc)

            self.config["trainResult"]["time"] = datetime.now()

        elif whichModel in ["vanilla_dsvdd","smoothed_dsvdd",'denoising_dsvdd']:
            from pprint import pprint

            for i in range(10):
                pprint(trainResult)
                pprint("")
                pprint("")
                pprint("")
            self.config["trainResult"] = {}
            
            self.config["trainResult"]["ae_trian_loss"] = {
                f"epoch_{idx}": str(i)
                for idx, i in enumerate(trainResult["ae_train_loss"])
            }
            self.config["trainResult"]["model_train_loss"] = {
                f"epoch_{idx}": str(i)
                for idx, i in enumerate(trainResult["model_train_loss"])
            }

            self.config["trainResult"]["average_precision"] = {
                f"epoch_{idx}": str(i)
                for idx, i in enumerate(trainResult["val_average_precision"])
            }
            self.config["trainResult"]["roc_auc"] = {
                f"epoch_{idx}": str(i)
                for idx, i in enumerate(trainResult["val_roc_auc"])
            }

            # self.config['trainResult']['confusion'] = trainResult['val_f1'][-1]

            confusions = trainResult["val_f1"]
            self.config["trainResult"]["val_result_changes"] = {}
            for idx, i in enumerate(confusions):
                self.config["trainResult"]["val_result_changes"][f"epoch_{idx}"] = {
                    "TN": str(i[1]),
                    "FP": str(i[2]),
                    "FN": str(i[3]),
                    "TP": str(i[4]),
                    "precision": str(i[5]),
                    "recall": str(i[6]),
                    "f1_score": str(i[7]),
                }

            self.config["trainResult"]["time"] = datetime.now()

    def runTestLoop(self):

        testDataSet = myNewDataset(configs=self.config, isTrain=False)

        whichModel = self.config["which_model"]

        if whichModel == "ssvdd":
            model = sSVDDLoop(config=self.config)

        elif whichModel == "ocsvm":
            model = ocsvmLoop(config=self.config)
        elif whichModel == "vanilla_dsvdd":
            model = vanillaDsvddLoop(config=self.config)

        loadedModel = model.load_model()

        testResult = model.runTest(dataSet=testDataSet, loadedModel=loadedModel)

        print("test complete")

        return testResult

    def testResultToConfig(self, testResult):

        whichModel = self.config["which_model"]
        if whichModel in ["ssvdd", "ocsvm"]:

            y_pred = testResult["y_pred"]
            y_anomaly_score = testResult["y_anomaly_score"]
            y_test = testResult["y_test"]

            y_pred = np.where(y_pred == -1, 0, 1)
            y_test = np.where(y_test == -1, 0, 1)

            tn, fp, fn, tp = confusion_matrix(y_pred=y_pred, y_true=y_test).ravel()

            precisionScore = precision_score(y_pred=y_pred, y_true=y_test)

            recallScore = recall_score(y_pred=y_pred, y_true=y_test)

            f1Score = f1_score(y_pred=y_pred, y_true=y_test)

            averagePrecision = average_precision_score(
                y_score=y_anomaly_score, y_true=y_test
            )

            rocAuc = roc_auc_score(y_score=y_anomaly_score, y_true=y_test)

            self.config["testResult"] = {}

            self.config["testResult"]["truePos"] = str(tp)
            self.config["testResult"]["trueNeg"] = str(tn)
            self.config["testResult"]["falsePos"] = str(fp)
            self.config["testResult"]["falseNeg"] = str(fn)

            self.config["testResult"]["precision"] = str(precisionScore)
            self.config["testResult"]["recall"] = str(recallScore)
            self.config["testResult"]["f1"] = str(f1Score)

            self.config["testResult"]["averagePrecision"] = str(averagePrecision)
            self.config["testResult"]["rocAuc"] = str(rocAuc)

            self.config["testResult"]["time"] = datetime.now()

        elif whichModel == "vanilla_dsvdd":
            self.config["testResult"] = {}
            self.config["testResult"]["average_precision"] = str(
                testResult["test_average_precision"][0]
            )
            self.config["testResult"]["roc_auc"] = str(testResult["test_roc_auc"][0])

            f_scores = testResult["test_f1"][0]
            self.config["testResult"]["threshold"] = str(f_scores[0])
            self.config["testResult"]["TN"] = str(f_scores[1])
            self.config["testResult"]["FP"] = str(f_scores[2])
            self.config["testResult"]["FN"] = str(f_scores[3])
            self.config["testResult"]["TP"] = str(f_scores[4])
            self.config["testResult"]["precision"] = str(f_scores[5])
            self.config["testResult"]["recall"] = str(f_scores[6])
            self.config["testResult"]["f1_score"] = str(f_scores[7])
            self.config["testResult"]["time"] = datetime.now()
            # from pprint import pprint
            # for i in range(10):
            #     pprint(self.config['testResult'])
