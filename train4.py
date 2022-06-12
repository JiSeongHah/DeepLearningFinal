from torch.optim import AdamW, Adam,SGD
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,Dataset,TensorDataset
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.svm import SVC
import pickle
from torch.nn import Parameter
import math
from sklearn.neighbors import KNeighborsClassifier


##############################UTILITY Funcs ad Models ####################################
##############################UTILITY Funcs ad Models ####################################
##############################UTILITY Funcs ad Models ####################################
class simpleDNN(nn.Module):

    def __init__(self,
                 innerNum1=64,
                 innerNum2=64,
                 innerNum3=64,
                 outNum=2):
        super(simpleDNN, self).__init__()


        self.innerNum1 = innerNum1
        self.innerNum2 = innerNum2
        self.innerNum3 = innerNum3
        self.outNum = outNum

        self.lin1 = nn.Linear(in_features=17,out_features=self.innerNum1)
        self.lin2 = nn.Linear(in_features=self.innerNum1,out_features=self.innerNum2)
        self.lin3 = nn.Linear(in_features=self.innerNum2, out_features=self.innerNum3)
        self.lin4 = nn.Linear(in_features=self.innerNum3, out_features=self.outNum)

    def forward(self, x):
        out = F.relu(self.lin1(x))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = F.normalize(self.lin4(out),dim=1)

        return out

class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature=17, out_feature=2, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        #print(cosine.size())
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.clamp(torch.pow(cosine, 2),max=1) + 1e-6)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #print(phi.size())

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        #print(one_hot)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def mk_name(*args,**name_value_dict):
    total_name = ''
    additional_arg = ''

    for arg in args:
        additional_arg += (str(arg)+'_')

    for name_value in name_value_dict.items():
        name = name_value[0]
        value = name_value[1]
        total_name += (str(name)+str(value)+'_')

    total_name += additional_arg[:-1]

    return total_name

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'making {str(directory)} complete successfully!')
    except OSError:
        print("Error: Failed to create the directory.")


def calAvgError(lossLst,windowNum=10):
    x = np.asarray(lossLst[-windowNum:])
    y = np.asarray(lossLst[-windowNum-1:-1])

    error = np.mean(abs(x-y))

    return error

##############################UTILITY Funcs ad Models ####################################
##############################UTILITY Funcs ad Models ####################################
##############################UTILITY Funcs ad Models ####################################








####################################Deep Learning Model###################################
####################################Deep Learning Model###################################
####################################Deep Learning Model###################################

class MyFinalprojectPredictor(nn.Module):
    def __init__(self,
                 baseDir,
                 plotSaveDir,
                 whichModel,
                 whichLoss,
                 innerNum,
                 trnBSize=128,
                 valBSize=128,
                 stopThreshold=1e-3,
                 wDecay=0.0,
                 lossWeight=torch.tensor([0.22,0.78]),
                 lr=3e-4,
                 iterToAccumul=2,
                 dataScale=True,
                 optim='adam',
                 gpuUse= True
                 ):
        super(MyFinalprojectPredictor,self).__init__()


        self.baseDir = baseDir
        self.plotSaveDir = plotSaveDir
        self.whichModel = whichModel
        self.whichLoss = whichLoss
        self.optim = optim
        self.gpuUse = gpuUse
        self.lr = lr
        self.wDecay = wDecay
        self.lossWeight = lossWeight
        self.stopThreshold = stopThreshold

        self.innerNum = innerNum

        self.trnBSize = trnBSize
        self.valBSize = valBSize

        self.iterToAccumul = iterToAccumul
        self.dataScale = dataScale

        if self.whichModel == 'simpleDNN':
            self.model = simpleDNN(innerNum1=self.innerNum,
                                   innerNum2=self.innerNum,
                                   innerNum3=self.innerNum)

        ##### use GPU or not ###########################
        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)
        ##### use GPU or not ###########################

        ########## Optim config ##################
        if self.optim == 'adam':
            self.optimizer = Adam(self.model.parameters(),
                                  lr=self.lr,  # 학습률
                                  eps = 1e-9,
                                  weight_decay=self.wDecay
                                    # 0으로 나누는 것을 방지하기 위한 epsilon 값
                                  )
        elif self.optim == 'sgd':
            self.optimizer = SGD(self.model.parameters(),
                                  lr=1e-3,  # 학습률
                                  )
        else:
            self.optimizer = AdamW(self.model.parameters(),
                                  lr=self.lr,  # 학습률
                                   weight_decay=0.01
                                  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                                  )
        ########## Optim config ##################

        ########### loss function config #############
        if self.whichLoss == 'weightFocal':
            self.lossMethod = FocalLoss(weight=self.lossWeight)
        elif self.whichLoss == 'vanillaFocal':
            self.lossMethod = FocalLoss()
        else:
            self.lossMethod = nn.CrossEntropyLoss()

        ########### loss function config #############

        ########### train and validation Data config#######################
        trainDataBefSplit = np.loadtxt(self.baseDir+'train_input.txt',delimiter=',')
        trainLabelBefSplit = np.loadtxt(self.baseDir+'train_target.txt')
        if self.dataScale == True:
            scaler = preprocessing.StandardScaler().fit(trainDataBefSplit)
            trainDataBefSplit = scaler.transform(trainDataBefSplit)


        trnInput,valInput,trnLabel,valLabel = train_test_split(trainDataBefSplit,trainLabelBefSplit,
                                                                test_size=0.1,shuffle=True)

        print(trnInput.shape,trnLabel.shape,valInput.shape,valLabel.shape)

        trnDataset = TensorDataset(torch.from_numpy(trnInput),torch.from_numpy(trnLabel))
        self.trnDataloader = DataLoader(trnDataset,batch_size=self.trnBSize,shuffle=True)

        valDataset = TensorDataset(torch.from_numpy(valInput), torch.from_numpy(valLabel))
        self.valDataloader = DataLoader(valDataset, batch_size=self.valBSize,shuffle=False)
        ########### train and validation Data config#######################

        self.trnLossLst = []
        self.trnLossLstAvg = []
        self.valLossLst = []
        self.valLossLstAvg = []

        self.trnAccLst = []
        self.trnAccLstAvg = []
        self.valAccLst = []
        self.valAccLstAvg = []

        self.model.to(self.device)


    def forward(self,x):
        out = self.model(x)

        return out

    def getLoss(self,pred,label):

        predArgMax = torch.argmax(pred,dim=1)

        ######### one is positive, zero is negative ##########
        labelZero = label == 0
        labelOne = label == 1

        predZero = predArgMax == 0
        predOne = predArgMax == 1

        TP = torch.sum((labelOne == predOne).float()).item()
        TN = torch.sum((labelZero == predZero).float()).item()
        FP = torch.sum((labelZero == predOne).float()).item()
        FN = torch.sum((labelOne == predZero).float()).item()

        oneAcc = TP/(TP+FN)
        zeroAcc = TN/(TN+FP)

        balancedAcc = 0.5 * (oneAcc + zeroAcc)

        return self.lossMethod(pred,label), balancedAcc


    def flushLst(self):

        self.trnLossLst.clear()
        self.valLossLst.clear()

        self.trnAccLst.clear()
        self.valAccLst.clear()
        print('flushing complete!')

    def trainingStep(self):

        self.model.train()
        self.optimizer.zero_grad()

        TDataloader = tqdm(self.trnDataloader)
        globalTime = time.time()

        with torch.set_grad_enabled(True):
            for idx,(inputs,labels) in enumerate(TDataloader):
                localTime = time.time()

                inputs= inputs.float()
                labels = labels.long()
                inputs = inputs.to(self.device)
                logits = self.forward(inputs).cpu()

                loss,acc = self.getLoss(pred=logits,label=labels)
                ResultLoss = loss/self.iterToAccumul

                ResultLoss.backward()

                self.trnLossLst.append(loss.item())
                self.trnAccLst.append(acc)

                if (idx +1) % self.iterToAccumul ==0:

                    self.optimizer.step()
                    self.optimizer.zero_grad()


                localTimeElaps = round(time.time() - localTime,2)
                globalTimeElaps = round(time.time() - globalTime,2)

                TDataloader.set_description(f'Processing : {idx} of {len(TDataloader)}')
                TDataloader.set_postfix(Gelapsed=globalTimeElaps,
                                        Lelapsed=localTimeElaps,
                                        loss=loss.item())


        self.model.eval()

    def validationStep(self):

        self.model.eval()
        self.optimizer.zero_grad()

        VDataloader = tqdm(self.valDataloader)
        globalTime = time.time()

        with torch.set_grad_enabled(False):
            for idx, (inputs, labels) in enumerate(VDataloader):
                localTime = time.time()

                inputs = inputs.float()
                labels = labels.long()
                inputs = inputs.to(self.device)
                logits = self.forward(inputs).cpu()

                loss, acc = self.getLoss(pred=logits, label=labels)
                ResultLoss = loss / self.iterToAccumul

                self.valLossLst.append(loss.item())
                self.valAccLst.append(acc)

                localTimeElaps = round(time.time() - localTime, 2)
                globalTimeElaps = round(time.time() - globalTime, 2)

                VDataloader.set_description(f'Processing : {idx} of {len(VDataloader)}')
                VDataloader.set_postfix(Gelapsed=globalTimeElaps,
                                        Lelapsed=localTimeElaps,
                                        loss=loss.item())

        self.model.train()


    def validationStepEnd(self):

        self.trnLossLstAvg.append(np.mean(self.trnLossLst))
        self.trnAccLstAvg.append(np.mean(self.trnAccLst))
        self.valLossLstAvg.append(np.mean(self.valLossLst))
        self.valAccLstAvg.append(np.mean(self.valAccLst))

        fig = plt.figure(constrained_layout=True)
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.plot(range(len(self.trnLossLstAvg)), self.trnLossLstAvg)
        ax1.set_title('train loss')
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.plot(range(len(self.trnAccLstAvg)), self.trnAccLstAvg)
        ax2.set_title('train acc')

        ax3 = fig.add_subplot(1, 4, 3)
        ax3.plot(range(len(self.valLossLstAvg)), self.valLossLstAvg)
        ax3.set_title('val loss')
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.plot(range(len(self.valAccLstAvg)), self.valAccLstAvg)
        ax4.set_title('val acc ')

        plt.savefig(self.plotSaveDir + 'Result.png', dpi=200)
        print('saving plot complete!')
        plt.close()
        plt.cla()
        plt.clf()

        self.flushLst()
        # print(len(self.trnLossLstAvg))
        # print(len(self.trnAccLstAvg))
        # print(len(self.valLossLstAvg))
        # print(len(self.valAccLstAvg))



    def START_TRN_VAL(self,iterNum):

        for i in range(iterNum):

            self.trainingStep()
            self.validationStep()
            self.validationStepEnd()

            if len(self.valAccLstAvg) > 11:
                if calAvgError(self.valAccLstAvg) <= self.stopThreshold:
                    print(f'avgError : {calAvgError(self.valAccLstAvg)} is less than'
                          f'Threshold :  {self.stopThreshold} so breaking now...')
                    break

####################################Deep Learning Model###################################
####################################Deep Learning Model###################################
####################################Deep Learning Model###################################




#################################### SVM Version #########################################
#################################### SVM Version #########################################
#################################### SVM Version #########################################

class MyFinalSVMClassifier():

    def __init__(self,
                 baseDir,
                 modelPlotSaveDir,
                 C,
                 kernel,
                 gamma,
                 dataScale=True
                 ):

        self.baseDir = baseDir
        self.modelPlotSaveDir = modelPlotSaveDir
        self.dataScale = dataScale

        self.C = C
        self.kernel = kernel
        self.gamma = gamma

        ########### train and validation Data config#######################
        trainDataBefSplit = np.loadtxt(self.baseDir+'train_input.txt',delimiter=',')
        trainLabelBefSplit = np.loadtxt(self.baseDir+'train_target.txt')
        if self.dataScale == True:
            scaler = preprocessing.StandardScaler().fit(trainDataBefSplit)
            trainDataBefSplit = scaler.transform(trainDataBefSplit)


        self.trnInput,\
        self.valInput,\
        self.trnLabel,\
        self.valLabel = train_test_split(trainDataBefSplit,
                                         trainLabelBefSplit,
                                         test_size=0.1,
                                         shuffle=True)
        ########### train and validation Data config#######################


        ############ test Data Config ###################################
        testData = np.loadtxt(self.baseDir+'test_input.txt',delimiter=',')
        if self.dataScale == True:
            scaler = preprocessing.StandardScaler().fit(testData)
            self.testData = scaler.transform(testData)
        ############ test Data Config ###################################

    def getAcc(self,pred,label):

        ######### one is positive, zero is negative ##########
        labelZero = label == 0
        labelOne = label == 1

        predZero = pred == 0
        predOne = pred == 1

        TP = np.sum((labelOne == predOne)*1.0)
        TN = np.sum((labelZero == predZero)*1.0)
        FP = np.sum((labelZero == predOne)*1.0)
        FN = np.sum((labelOne == predZero)*1.0)

        oneAcc = TP/(TP+FN)
        zeroAcc = TN/(TN+FP)

        balancedAcc = 0.5 * (oneAcc + zeroAcc)

        return balancedAcc


    def trainAndSaveSVC(self):

        clf = SVC(C=self.C,
                  kernel=self.kernel,
                  gamma=self.gamma,
                  verbose=True)

        clf.fit(self.trnInput,self.trnLabel)

        pred = clf.predict(self.valInput)

        acc = self.getAcc(pred,self.valLabel)

        for i in range(5):
            print(acc)

        with open(self.modelPlotSaveDir+'savedModel.pkl','wb') as ff:
            pickle.dump(clf,ff)

        print(f'SVC training with C : {self.C}, kernel : {self.kernel}, gamma : {self.gamma} complete!!!')
        print('')
        print('')
        print('')

#################################### SVM Version #########################################
#################################### SVM Version #########################################
#################################### SVM Version #########################################






################################### Arcface with KNN version ##############################
################################### Arcface with KNN version ##############################
################################### Arcface with KNN version ##############################
class MyFinalArcface(nn.Module):
    def __init__(self,
                 baseDir,
                 modelPlotSaveDir,
                 lossMethod,
                 s,
                 m,
                 innerNum,
                 dataScale=True,
                 stopThreshold=1e-3,
                 gpuUse= True,
                 iter_to_accumul=2,
                 topkNum=17,
                 bSizeTrn= 64,
                 bSizeVal=64,
                 lr=3e-4,
                 eps=1e-9):


        super(MyFinalArcface,self).__init__()



        self.baseDir = baseDir
        self.modelPlotSaveDir = modelPlotSaveDir
        self.iter_to_accumul = iter_to_accumul

        self.gpuUse = gpuUse

        self.lr = lr
        self.eps = eps
        self.bSizeTrn = bSizeTrn
        self.bSizeVal = bSizeVal
        self.lossMethod = lossMethod

        self.s = s
        self.m = m
        self.topkNum = topkNum
        self.stopThreshold = stopThreshold
        self.dataScale = dataScale
        self.innerNum = innerNum


        self.loss_lst_trn = []
        self.loss_lst_trn_tmp = []
        self.loss_lst_val = []
        self.loss_lst_val_tmp = []

        self.acc_lst_trn = []
        self.acc_lst_trn_tmp = []
        self.acc_lst_val = []
        self.acc_lst_val_tmp = []



        ###################MODEL SETTING###########################
        print('failed loading model, loaded fresh model')
        self.MyBackbone = simpleDNN(innerNum1=self.innerNum,
                                    innerNum2=self.innerNum,
                                    innerNum3=self.innerNum,
                                    outNum=17
                                    )
        self.MyArcProduct = ArcMarginProduct(in_feature=17,
                                             out_feature=2,
                                             s=self.s,
                                             m=self.m)

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)


        self.optimizer = AdamW([{'params':self.MyBackbone.parameters()},
                              {'params':self.MyArcProduct.parameters()}],
                              lr=self.lr  # 학습률
                                # 0으로 나누는 것을 방지하기 위한 epsilon 값
                              )

        ########### train and validation Data config#######################
        trainDataBefSplit = np.loadtxt(self.baseDir+'train_input.txt',delimiter=',')
        trainLabelBefSplit = np.loadtxt(self.baseDir+'train_target.txt')
        if self.dataScale == True:
            scaler = preprocessing.StandardScaler().fit(trainDataBefSplit)
            trainDataBefSplit = scaler.transform(trainDataBefSplit)


        trnInput,valInput,trnLabel,valLabel = train_test_split(trainDataBefSplit,trainLabelBefSplit,
                                                                test_size=0.1,shuffle=True)


        trnDataset = TensorDataset(torch.from_numpy(trnInput),torch.from_numpy(trnLabel))
        self.trnDataloader = DataLoader(trnDataset,batch_size=self.bSizeTrn,shuffle=True)

        valDataset = TensorDataset(torch.from_numpy(valInput), torch.from_numpy(valLabel))
        self.valDataloader = DataLoader(valDataset, batch_size=self.bSizeVal,shuffle=False)
        ########### train and validation Data config#######################

        WEIGHT = torch.tensor([0.22, 0.78])

        if self.lossMethod == 'weightFocal':
            self.loss = FocalLoss(weight=WEIGHT)
        elif self.lossMethod == 'vanillaFocal':
            self.loss = FocalLoss()
        elif self.lossMethod == 'weighCross':
            self.loss = nn.CrossEntropyLoss(weight=WEIGHT)
        else:
            self.loss = nn.CrossEntropyLoss()

        self.MyBackbone.to(device=self.device)
        self.MyArcProduct.to(device=self.device)


    def forward(self,x):

        out = self.MyBackbone(x)

        return out

    def forwardArc(self,x,label):

        out = self.MyArcProduct(x,label)

        return out

    def calLoss(self,logit,label):

        loss = nn.CrossEntropyLoss()

        LOSS = loss(logit,label)

        pred = torch.argmax(logit.clone().detach(),dim=1)

        ######### one is positive, zero is negative ##########
        labelZero = label == 0
        labelOne = label == 1

        predZero = pred == 0
        predOne = pred == 1

        TP = torch.sum((labelOne == predOne).float()).item()
        TN = torch.sum((labelZero == predZero).float()).item()
        FP = torch.sum((labelZero == predOne).float()).item()
        FN = torch.sum((labelOne == predZero).float()).item()

        oneAcc = TP/(TP+FN)
        zeroAcc = TN/(TN+FP)

        balancedAcc = 0.5 * (oneAcc + zeroAcc)

        return LOSS, balancedAcc

    def calAcc(self,pred,label):

        ######### one is positive, zero is negative ##########
        labelZero = label == 0
        labelOne = label == 1

        predZero = pred == 0
        predOne = pred == 1

        TP = torch.sum((labelOne == predOne).float()).item()
        TN = torch.sum((labelZero == predZero).float()).item()
        FP = torch.sum((labelZero == predOne).float()).item()
        FN = torch.sum((labelOne == predZero).float()).item()

        oneAcc = TP / (TP + FN)
        zeroAcc = TN / (TN + FP)

        balancedAcc = 0.5 * (oneAcc + zeroAcc)

        return balancedAcc

    def trainingStep(self):

        torch.autograd.set_detect_anomaly(True)

        self.MyBackbone.train()
        self.MyArcProduct.train()

        self.optimizer.zero_grad()
        TDataloader = tqdm(self.trnDataloader)

        with torch.set_grad_enabled(True):
            globalTime= time.time()

            for idx, (bInput, bLabel)  in enumerate(TDataloader):
                localTime= time.time()

                bInput = bInput.float()
                bLabel = bLabel.long()
                bInput = bInput.to(self.device)
                bLabel = bLabel.to(self.device)

                bFeature = self.forward(bInput)
                bLogit = self.forwardArc(bFeature,bLabel)

                bLogit = bLogit.cpu()
                bLabel = bLabel.cpu()

                ResultLoss,ResultAcc = self.calLoss(bLogit,bLabel)
                ResultLoss = ResultLoss/self.iter_to_accumul

                ResultLoss.backward()

                self.loss_lst_trn_tmp.append(ResultLoss.item())
                self.acc_lst_trn_tmp.append(ResultAcc)

                if (idx + 1) % self.iter_to_accumul == 0:
                    # nn.utils.clip_grad_norm_(self.MyBackbone.parameters(),
                    #                         max_norm=1.0)
                    # nn.utils.clip_grad_norm_(self.MyArcProduct.parameters(),
                    #                         max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                localTimeElaps = round(time.time() - localTime,2)
                globalTimeElaps = round(time.time() - globalTime,2)

                TDataloader.set_description(f'Processing : {idx} of {len(TDataloader)}')
                TDataloader.set_postfix(Gelapsed=globalTimeElaps,
                                        Lelapsed=localTimeElaps,
                                        loss=ResultLoss.item(),
                                        acc=ResultAcc)


        self.MyBackbone.eval()
        self.MyArcProduct.eval()

    def valdatingStep(self):

        self.MyBackbone.eval()
        self.MyArcProduct.eval()
        self.optimizer.zero_grad()

        TDataloader = tqdm(self.trnDataloader)
        VDataloader = tqdm(self.valDataloader)

        criterionFeatures = []
        criterionLabels = []

        with torch.set_grad_enabled(False):
            for idx, (bInput,bLabel) in enumerate(TDataloader):

                bLabel = bLabel.long()
                bInput = bInput.float()
                bInput = bInput.to(self.device)

                bFeature = self.forward(bInput).cpu()


                criterionFeatures.append(bFeature.clone().detach())
                criterionLabels.append(bLabel)

        criterionFeatures = torch.cat(criterionFeatures).numpy()
        criterionLabels = torch.cat(criterionLabels).numpy()

        knnClf = KNeighborsClassifier(n_neighbors=self.topkNum)

        compareFeatures = []
        compareGTLabels = []

        with torch.set_grad_enabled(False):
            for idx, (valBInput, valBLabel) in enumerate(VDataloader):

                valBInput =valBInput.float()
                valBInput = valBInput.to(self.device)
                valBFeature = self.forward(valBInput).cpu()

                compareFeatures.append(valBFeature.clone().detach())
                compareGTLabels.append(valBLabel)

        compareFeatures = torch.cat(compareFeatures).numpy()
        compareGTLabels = torch.cat(compareGTLabels).numpy()

        print(criterionFeatures.shape,criterionLabels.shape)
        print(compareFeatures.shape,compareGTLabels.shape)
        # print(compareFeatures,compareGTLabels)


        knnClf.fit(criterionFeatures,criterionLabels)
        predictedLabels = knnClf.predict(compareFeatures)

        valAcc = self.calAcc(pred=torch.from_numpy(predictedLabels),
                             label=torch.from_numpy(compareGTLabels))

        self.acc_lst_val_tmp.append(valAcc)

        torch.set_grad_enabled(True)
        self.MyBackbone.train()
        self.MyArcProduct.train()

    def valdatingStepEnd(self):

        self.loss_lst_trn.append(self.iter_to_accumul * np.mean(self.loss_lst_trn_tmp))
        self.acc_lst_trn.append(np.mean(self.acc_lst_trn_tmp))
        self.acc_lst_val.append(np.mean(self.acc_lst_val_tmp))

        fig = plt.figure(constrained_layout=True)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(range(len(self.loss_lst_trn)), self.loss_lst_trn)
        ax1.set_title('train loss')

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(range(len(self.acc_lst_trn)), self.acc_lst_trn)
        ax2.set_title('train acc')

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(range(len(self.acc_lst_val)), self.acc_lst_val)
        ax3.set_title('val acc')

        plt.savefig(self.modelPlotSaveDir + 'Result.png', dpi=200)
        print('saving plot complete!')
        plt.close()
        plt.cla()
        plt.clf()

        self.flushLst()

    def flushLst(self):

        self.loss_lst_trn_tmp.clear()
        self.acc_lst_trn_tmp.clear()
        self.acc_lst_val_tmp.clear()

    def START_TRN_VAL(self,iterNum):


        for i in range(iterNum):

            self.trainingStep()
            self.valdatingStep()
            self.valdatingStepEnd()

            if len(self.loss_lst_val) > 11:
                if calAvgError(self.loss_lst_val) <= self.stopThreshold:
                    print(f'avgError : {calAvgError(self.loss_lst_val)} is less than'
                          f'Threshold :  {self.stopThreshold} so breaking now...')
                    break


if __name__ == '__main__':


    # ###################SVC TRAIN###################################
    # ###################SVC TRAIN###################################
    # ###################SVC TRAIN###################################
    # baseDir = '/home/a286winteriscoming/Downloads/FinalHomwork/'
    # CLst = [0.01,0.1,1,10,100]
    # gammaLst = [0.001,0.01,0.1,1,10]
    # kernelLst = ['rbf','poly','sigmoid']
    #
    # for kernel in kernelLst:
    #     for C in CLst:
    #         for gamma in gammaLst:
    #
    #             specificDirName = mk_name(dir='SVC/',
    #                                       kernel=kernel,
    #                                       C=C,
    #                                       gamma=gamma)
    #
    #             plotSaveDir = baseDir+specificDirName +'/'
    #             createDirectory(plotSaveDir)
    #
    #             doProject = MyFinalSVMClassifier(baseDir=baseDir,
    #                                              modelPlotSaveDir=plotSaveDir,
    #                                              C=C,
    #                                              kernel='rbf',
    #                                              gamma=gamma)
    #
    #             doProject.trainAndSaveSVC()
    # ###################SVC TRAIN###################################
    # ###################SVC TRAIN###################################
    # ###################SVC TRAIN###################################

    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    #######################DNN TRAIN #############################
    #######################DNN TRAIN #############################
    #######################DNN TRAIN #############################
    baseDir = '/home/a286/hjs_dir1/DeepLearningFinal/'
    whichModel = 'simpleDNN'
    optimLst = ['adam','sgd','adamw']
    innerNumLst = [64,128,256]
    lossLst = ['weightFocal','vanillaFocal','normalCE']


    for optim in optimLst:
        for innerNum in innerNumLst:
            for whichLoss in lossLst:
                specificDirName = mk_name(dir='DNN/',
                                          optim=optim,
                                          innerNum=innerNum,
                                          loss=whichLoss)

                plotSaveDir = baseDir+specificDirName +'/'
                createDirectory(plotSaveDir)
                doProject = MyFinalprojectPredictor(baseDir=baseDir,
                                                    plotSaveDir=plotSaveDir,
                                                    whichModel=whichModel,
                                                    innerNum=innerNum,
                                                    optim=optim,
                                                    whichLoss=whichLoss,
                                                    dataScale=True)

                startTrain = doProject.START_TRN_VAL(iterNum=100)
                torch.save(doProject,plotSaveDir+'models.pt')
    #######################DNN TRAIN #############################
    #######################DNN TRAIN #############################
    #######################DNN TRAIN #############################


    #######################Arcface TRAIN ########################
    #######################Arcface TRAIN ########################
    #######################Arcface TRAIN ########################

    baseDir = '/home/a286winteriscoming/Downloads/FinalHomwork/'
    whichModel = 'simpleDNN'
    optim = 'adam'
    innerNumLst = [64,128,256]
    sLst = [16,32,64]
    mLst = [0.4,0.5,0.6]
    whichLoss = 'normalCE'

    for innerNum in innerNumLst:
        for s in sLst:
            for m in mLst:
                specificDirName = mk_name(dir='arcface/',
                                          innerNum=innerNum,
                                          s=s,
                                          m=m)

                plotSaveDir = baseDir+specificDirName +'/'
                createDirectory(plotSaveDir)
                doProject = MyFinalArcface(baseDir=baseDir,
                                           modelPlotSaveDir=plotSaveDir,
                                           lossMethod=whichLoss,
                                           innerNum=innerNum,
                                           s=s,
                                           m=m
                                           )

                startTrain = doProject.START_TRN_VAL(iterNum=100)
                torch.save(doProject,plotSaveDir+'models.pt')


    #######################Arcface TRAIN ########################
    #######################Arcface TRAIN ########################
    #######################Arcface TRAIN ########################

























