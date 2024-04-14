import csv
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from scipy.stats import kde
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from vanillaDSVDD_model import naiveFCN, naivePreAutoEncoder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import copy
import os

from data_utils import change_data,check_and_normalize




class vanillaDsvddLoop:
    def __init__(self, config) -> None:
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

        self.AElossLstTrnTmp = []
        self.AElossLstTrn = []

        self.modelLossLstTrnTmp = []
        self.modelLossLstTrn = []

        self.val_averagePrecisionLst = []
        self.val_rocAucLst = []
        self.val_f_lst = []

        self.test_averagePrecisionLst = []
        self.test_rocAucLst = []
        self.test_f_lst = []

    def runTrain(self, dataSet):

        USE_CUDA = torch.cuda.is_available()
        print(USE_CUDA)

        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        print("학습을 진행하는 기기:", self.device)

        self.DSVDD_model = naiveFCN(
            hDim1=self.config["hDim1"],
            hDim2=self.config["hDim2"],
            hDim3=self.config["hDim3"],
            FVSize=self.config["FVSize"],
            inputSize=self.config["inputSize"],
        )

        self.DSVDD_preAE = naivePreAutoEncoder(
            hDim1=self.config["hDim1"],
            hDim2=self.config["hDim2"],
            hDim3=self.config["hDim3"],
            FVSize=self.config["FVSize"],
            inputSize=self.config["inputSize"],
        )

        self.Modeloptim = AdamW(
            self.DSVDD_model.parameters(), lr=3e-4, weight_decay=0.5e-6
        )

        self.AEoptim = AdamW(
            self.DSVDD_preAE.parameters(), lr=3e-4, weight_decay=0.5e-3
        )
        
        x_train_total, y_train_total = change_data(dataSet=dataSet,config=self.config)

        x_train_total = check_and_normalize(x_train_total,self.config,mode='train')
        
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_total, y_train_total, test_size=0.2, random_state=42
        )
        
        for eachEpoch in range(self.config["preAE_epoch"]):
            self.trainPreAE(
                aeTrainDataSet=(copy.deepcopy(x_train), copy.deepcopy(y_train))
            )
            self.trainPreAEEnd()

        self.saveWeightAE(iterNum=self.config["preAE_epoch"])
        self.transferAEtoMainModel()

        centre = self.setCentre(normalDataSet=(x_train, y_train))

        for eachEpoch in range(self.config["mainModel_epoch"]):
            self.trainMainModel(
                mainModelTrainDataSet=(copy.deepcopy(x_train), copy.deepcopy(y_train))
            )
            self.trainModelEnd()
            self.validationStep(validationDataset=(x_val, y_val))
            self.validationStepEnd()

        self.saveWeightMainModel(iterNum=self.config["mainModel_epoch"])

        save_dict = {
            "ae_train_loss": self.AElossLstTrn,
            "model_train_loss": self.modelLossLstTrn,
            "val_average_precision": self.val_averagePrecisionLst,
            "val_roc_auc": self.val_rocAucLst,
            "val_f1": self.val_f_lst,
        }

        return save_dict

    def calMSELoss(self, output, label, reduction="mean"):

        loss = nn.MSELoss(reduction=reduction)

        return loss(output, label)

    def trainPreAE(self, aeTrainDataSet):

        x_train, y_train = aeTrainDataSet[0], aeTrainDataSet[1]

        # y_train = y_train[:,np.newaxis]

        aeTrainTensorDataSet = TensorDataset(
            torch.tensor(x_train), torch.tensor(y_train)
        )

        self.DSVDD_preAE.to(self.device)
        self.DSVDD_preAE.train()

        aeTrainDataloader = tqdm(
            DataLoader(
                aeTrainTensorDataSet,
                batch_size=self.config["preAE_batch_size"],
                shuffle=True,
                drop_last=True,
            )
        )

        with torch.set_grad_enabled(True):

            for idx, bInputLabel in enumerate(aeTrainDataloader):

                bInput, bLabel = bInputLabel

                self.AEoptim.zero_grad()

                answer = bInput.clone().detach().float()

                bOutput = self.DSVDD_preAE(bInput.float().to(self.device)).cpu()

                loss = self.calMSELoss(bOutput, answer)

                loss.backward()
                self.AEoptim.step()

                self.AElossLstTrnTmp.append(loss.item())

        self.DSVDD_preAE.to("cpu")
        self.DSVDD_preAE.eval()

    def trainPreAEEnd(self):

        self.AElossLstTrn.append(np.mean(self.AElossLstTrnTmp))

        plotSavePath = os.path.join(self.config["modelSavePath"], "plots")
        os.makedirs(plotSavePath, exist_ok=True)

        plt.plot(range(len(self.AElossLstTrn)), self.AElossLstTrn)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Auto Encoder Train Loss")
        plt.savefig(os.path.join(plotSavePath, "aeTrainLossPlot.png"), dpi=300)
        plt.cla()
        plt.clf()
        plt.close()

        self.AElossLstTrnTmp.clear()

    def saveWeightAE(self, iterNum):

        preAeSavePath = os.path.join(self.config["modelSavePath"], "models/pre_ae")
        os.makedirs(preAeSavePath, exist_ok=True)

        torch.save(
            self.DSVDD_preAE.state_dict(),
            os.path.join(preAeSavePath, "ae_" + str(iterNum)) + ".pt",
        )

        print("saving AE weight complete")

    def saveWeightMainModel(self, iterNum):

        mainModelSavePath = os.path.join(
            self.config["modelSavePath"], "models/main_model"
        )
        os.makedirs(mainModelSavePath, exist_ok=True)

        torch.save(
            self.DSVDD_model.state_dict(),
            os.path.join(mainModelSavePath, "mainModel_" + str(iterNum)) + ".pt",
        )

        print("saving MainModel weight complete")

    def transferAEtoMainModel(self):

        print("transferring weight of auto encoder to main model")

        self.DSVDD_model.load_state_dict(self.DSVDD_preAE.state_dict(), strict=False)
        print("transferring weight of auto encoder to main model complete !!!")

    def setCentre(self, normalDataSet):

        x_train, y_train = normalDataSet[0], normalDataSet[1]

        aeTrainTensorDataSet = TensorDataset(
            torch.tensor(x_train), torch.tensor(y_train)
        )

        self.DSVDD_preAE.to(self.device)
        self.DSVDD_preAE.eval()

        z_ = []

        tqdm._instances.clear()
        aeTrainDataloader = tqdm(
            DataLoader(
                aeTrainTensorDataSet,
                batch_size=self.config["preAE_batch_size"],
                shuffle=True,
                drop_last=True,
            )
        )

        with torch.set_grad_enabled(False):
            for idx, bInputLabel in enumerate(aeTrainDataloader):

                bInput, bLabel = bInputLabel

                self.AEoptim.zero_grad()

                answer = bInput.clone().detach().float()

                eachZ = self.DSVDD_preAE.doEncode(bInput.float().to(self.device)).cpu()

                z_.append(eachZ.clone().detach())

        z_ = torch.cat(z_)

        c = torch.mean(z_, dim=0)

        self.DSVDD_preAE.to("cpu")

        cSave = c.numpy()
        cSavePath = os.path.join(self.config["modelSavePath"], "models/center")
        os.makedirs(cSavePath, exist_ok=True)
        np.save(os.path.join(cSavePath, "cSave.npy"), cSave)
        tqdm._instances.clear()

        self.centre = c

        self.DSVDD_preAE.to("cpu")
        return c

    def trainMainModel(self, mainModelTrainDataSet):

        x_train, y_train = mainModelTrainDataSet[0], mainModelTrainDataSet[1]

        mainModelTrainTensorDataSet = TensorDataset(
            torch.tensor(x_train), torch.tensor(y_train)
        )

        self.DSVDD_preAE.to(self.device)
        self.DSVDD_preAE.eval()

        self.DSVDD_model.to(self.device)
        self.DSVDD_model.train()

        tqdm._instances.clear()
        theDloader = tqdm(
            DataLoader(
                mainModelTrainTensorDataSet,
                batch_size=self.config["mainmodel_batch_size"],
                shuffle=True,
                num_workers=0,
            ),
            position=0,
            leave=True,
        )

        for idx, bInputLabel in enumerate(theDloader):

            bInput, bLabel = bInputLabel

            self.Modeloptim.zero_grad()

            with torch.set_grad_enabled(True):
                bOutput = self.DSVDD_model(bInput.float().to(self.device)).cpu()

            loss = self.calMSELoss(bOutput, self.centre.repeat(bOutput.size(0), 1))

            loss.backward()
            self.Modeloptim.step()

            self.modelLossLstTrnTmp.append(loss.item())

        self.DSVDD_model.to("cpu")
        self.DSVDD_model.eval()

    def trainModelEnd(self):

        self.modelLossLstTrn.append(np.mean(self.modelLossLstTrnTmp))

        plotSavePath = os.path.join(self.config["modelSavePath"], "plots")
        os.makedirs(plotSavePath, exist_ok=True)

        plt.plot(range(len(self.modelLossLstTrn)), self.modelLossLstTrn)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"DSVDD model Train Loss")
        plt.savefig(os.path.join(plotSavePath, "mainModelTrainLossPlot.png"), dpi=300)
        plt.cla()
        plt.clf()
        plt.close()

        self.modelLossLstTrnTmp.clear()

    def validationStep(self, validationDataset):

        x_val, y_val = validationDataset[0], validationDataset[1]

        valTensorDataSet = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

        valDataloader = DataLoader(
            valTensorDataSet,
            batch_size=self.config["preAE_batch_size"],
            shuffle=False,
            drop_last=False,
        )

        totalScoreLstVal = []
        totalLabelLstVal = []

        self.DSVDD_model.to(self.device)
        self.DSVDD_model.eval()

        theDloader = tqdm(valDataloader, position=0, leave=True)

        with torch.set_grad_enabled(False):

            for idx, (totalBInput) in enumerate(theDloader):

                bInput, bLabel = totalBInput

                self.Modeloptim.zero_grad()

                bOutput = self.DSVDD_model(bInput.float().to(self.device)).cpu()

                eachScore = self.calMSELoss(
                    bOutput, self.centre.repeat(bOutput.size(0), 1), reduction="none"
                )

                useMax = self.config["useMax_when_val"]
                if useMax == True:
                    totalScoreLstVal.append(torch.amax(eachScore, dim=1))
                else:
                    totalScoreLstVal.append(torch.mean(eachScore, dim=1))

                totalLabelLstVal.append(bLabel)

        totalLabelTrue, totalScores = (
            torch.cat(totalLabelLstVal).numpy(),
            torch.cat(totalScoreLstVal).numpy(),
        )

        print(f"shape of label : {totalLabelTrue.shape}")
        print(f"shape of score : {totalScores.shape}")

        print(
            f"min Score is : {min(totalScores)} while max Score is : {max(totalScores)}"
        )
        saveMin = min(totalScores)
        saveMax = max(totalScores)
        minMaxedScore = (totalScores - min(totalScores)) / (
            max(totalScores) - min(totalScores)
        )

        averagePrecisionScore = average_precision_score(
            y_true=totalLabelTrue, y_score=minMaxedScore
        )
        rocAucScore = roc_auc_score(y_true=totalLabelTrue, y_score=minMaxedScore)

        fLst = []
        resultPerThresholdLst = []

        thresholdLst = [i / 1000 for i in range(1, 1000)]
        for eachThreshold in thresholdLst:

            labelPred = np.where(totalScores >= eachThreshold, 1, 0)

            tn, fp, fn, tp = confusion_matrix(
                y_true=totalLabelTrue, y_pred=labelPred
            ).ravel()

            precisionScore = precision_score(y_true=totalLabelTrue, y_pred=labelPred)
            recallScore = recall_score(y_true=totalLabelTrue, y_pred=labelPred)
            f1Score = f1_score(y_true=totalLabelTrue, y_pred=labelPred)
            fLst.append(f1Score)
            resultPerThresholdLst.append(
                [eachThreshold, tn, fp, fn, tp, precisionScore, recallScore, f1Score]
            )

            printAll = self.config["printAll"]
            if printAll == True:
                print(resultPerThresholdLst[-1])

        fMax = resultPerThresholdLst[fLst.index(max(fLst))]

        print("mission complete")

        self.DSVDD_model.to("cpu")

        print([averagePrecisionScore, rocAucScore, fMax[-1], fMax])

        self.val_averagePrecisionLst.append(averagePrecisionScore)
        self.val_rocAucLst.append(rocAucScore)
        self.val_f_lst.append(fMax)

    def validationStepEnd(self):

        plotSavePath = os.path.join(self.config["modelSavePath"], "plots")

        plt.plot(range(len(self.val_averagePrecisionLst)), self.val_averagePrecisionLst)
        plt.xlabel("Epoch")
        plt.ylabel("Average precision")
        plt.title(f"DSVDD model validation average precision")
        plt.savefig(os.path.join(plotSavePath, "valAveragePrecision.png"), dpi=300)
        plt.cla()
        plt.clf()
        plt.close()

        plt.plot(range(len(self.val_rocAucLst)), self.val_rocAucLst)
        plt.xlabel("Epoch")
        plt.ylabel("Roc auc")
        plt.title(f"DSVDD model validation roc auc")
        plt.savefig(os.path.join(plotSavePath, "valRocAuc.png"), dpi=300)
        plt.cla()
        plt.clf()
        plt.close()

        fOnlyLst = [i[-1] for i in self.val_f_lst]
        plt.plot(range(len(fOnlyLst)), fOnlyLst)
        plt.xlabel("Epoch")
        plt.ylabel("f1 score")
        plt.title(f"DSVDD model validation f1 score")
        plt.savefig(os.path.join(plotSavePath, "val_f1_score.png"), dpi=300)
        plt.cla()
        plt.clf()
        plt.close()

    def runTest(self, dataSet, loadedModel):

        USE_CUDA = torch.cuda.is_available()
        print(USE_CUDA)

        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        print("학습을 진행하는 기기:", self.device)

        self.DSVDD_model = loadedModel["main_model"]
        self.DSVDD_preAE = loadedModel["pre_ae"]

        self.Modeloptim = AdamW(
            self.DSVDD_model.parameters(), lr=3e-4, weight_decay=0.5e-6
        )

        self.AEoptim = AdamW(
            self.DSVDD_preAE.parameters(), lr=3e-4, weight_decay=0.5e-3
        )

        x_test, y_test = change_data(dataSet=dataSet,config=self.config)
        
        x_test = check_and_normalize(x_test,self.config,mode='test')
        
        cSavePath = os.path.join(self.config["modelSavePath"], "models/center")
        self.centre = torch.tensor(np.load(os.path.join(cSavePath, "cSave.npy")))

        self.testStep(testDataset=(x_test, y_test))
        self.testStepEnd()

        save_dict = {
            "test_average_precision": self.test_averagePrecisionLst,
            "test_roc_auc": self.test_rocAucLst,
            "test_f1": self.test_f_lst,
        }

        return save_dict

    def testStep(self, testDataset):

        x_test, y_test = testDataset[0], testDataset[1]

        testTensorDataSet = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

        testDataloader = DataLoader(
            testTensorDataSet,
            batch_size=self.config["mainmodel_batch_size"],
            shuffle=False,
            drop_last=False,
        )

        totalScoreLstTest = []
        totalLabelLstTest = []

        self.DSVDD_model.to(self.device)
        self.DSVDD_model.eval()

        theDloader = tqdm(testDataloader, position=0, leave=True)

        with torch.set_grad_enabled(False):

            for idx, (totalBInput) in enumerate(theDloader):

                bInput, bLabel = totalBInput

                self.Modeloptim.zero_grad()

                bOutput = self.DSVDD_model(bInput.float().to(self.device)).cpu()

                for i in range(10):
                    print(bOutput.size(), self.centre.shape)

                eachScore = self.calMSELoss(
                    bOutput, self.centre.repeat(bOutput.size(0), 1), reduction="none"
                )

                useMax = self.config["useMax_when_val"]
                if useMax == True:
                    totalScoreLstTest.append(torch.amax(eachScore, dim=1))
                else:
                    totalScoreLstTest.append(torch.mean(eachScore, dim=1))

                totalLabelLstTest.append(bLabel)

        totalLabelTrue, totalScores = (
            torch.cat(totalLabelLstTest).numpy(),
            torch.cat(totalScoreLstTest).numpy(),
        )

        print(f"shape of label : {totalLabelTrue.shape}")
        print(f"shape of score : {totalScores.shape}")

        print(
            f"min Score is : {min(totalScores)} while max Score is : {max(totalScores)}"
        )
        saveMin = min(totalScores)
        saveMax = max(totalScores)
        minMaxedScore = (totalScores - min(totalScores)) / (
            max(totalScores) - min(totalScores)
        )

        averagePrecisionScore = average_precision_score(
            y_true=totalLabelTrue, y_score=minMaxedScore
        )
        rocAucScore = roc_auc_score(y_true=totalLabelTrue, y_score=minMaxedScore)

        fLst = []
        resultPerThresholdLst = []

        thresholdLst = [i / 1000 for i in range(1, 1000)]
        for eachThreshold in thresholdLst:

            labelPred = np.where(totalScores >= eachThreshold, 1, 0)

            tn, fp, fn, tp = confusion_matrix(
                y_true=totalLabelTrue, y_pred=labelPred
            ).ravel()

            precisionScore = precision_score(y_true=totalLabelTrue, y_pred=labelPred)
            recallScore = recall_score(y_true=totalLabelTrue, y_pred=labelPred)
            f1Score = f1_score(y_true=totalLabelTrue, y_pred=labelPred)
            fLst.append(f1Score)
            resultPerThresholdLst.append(
                [eachThreshold, tn, fp, fn, tp, precisionScore, recallScore, f1Score]
            )

            printAll = self.config["printAll"]
            if printAll == True:
                print(resultPerThresholdLst[-1])

        fMax = resultPerThresholdLst[fLst.index(max(fLst))]

        print("mission complete")

        self.DSVDD_model.to("cpu")

        print([averagePrecisionScore, rocAucScore, fMax[-1], fMax])

        self.test_averagePrecisionLst.append(averagePrecisionScore)
        self.test_rocAucLst.append(rocAucScore)
        self.test_f_lst.append(fMax)

    def testStepEnd(self):

        pass

    def saveModel(self, trainResult):

        pass

    def load_model(self):

        aeIterNum = self.config["preAE_epoch"]
        mainModelIterNum = self.config["mainModel_epoch"]

        DSVDD_preAE = naivePreAutoEncoder(
            hDim1=self.config["hDim1"],
            hDim2=self.config["hDim2"],
            hDim3=self.config["hDim3"],
            FVSize=self.config["FVSize"],
            inputSize=self.config["inputSize"],
        )

        print(f"loading AE weight start...")
        preAeSavePath = os.path.join(self.config["modelSavePath"], "models/pre_ae")

        loadedAeWeight = torch.load(
            os.path.join(preAeSavePath, "ae_" + str(aeIterNum)) + ".pt"
        )

        missing = DSVDD_preAE.load_state_dict(loadedAeWeight)

        print("loading AE weight complete!")

        DSVDD_model = naiveFCN(
            hDim1=self.config["hDim1"],
            hDim2=self.config["hDim2"],
            hDim3=self.config["hDim3"],
            FVSize=self.config["FVSize"],
            inputSize=self.config["inputSize"],
        )

        print(f"loading main model weight start...")
        mainModelSavePath = os.path.join(
            self.config["modelSavePath"], "models/main_model"
        )

        loadedMainModelWight = torch.load(
            os.path.join(mainModelSavePath, "mainModel_" + str(mainModelIterNum))
            + ".pt"
        )
        missing = DSVDD_model.load_state_dict(loadedMainModelWight)
        print("saving MainModel weight complete!")

        return {"pre_ae": DSVDD_preAE, "main_model": DSVDD_model}
