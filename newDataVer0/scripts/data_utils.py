import torch
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


def touch_abnormal(dataSet, config, mode):

    assert mode in ["normal_only", "all"]

    if mode == "abnormal_only":

        whichLabelAbnormal = config["which_label_abnormal"]

        x_total = []
        y_total = []

        for eachData in dataSet:
            if eachData[1] == whichLabelAbnormal:
                x_total.append(eachData[0])
                y_total.append(1)

        x_total = np.stack(x_total)
        y_total = np.stack(y_total)

        return x_total, y_total

    else:
        x_total = []
        y_total = []

        for eachData in dataSet:
            x_total.append(eachData[0])
            y_total.append(eachData[1])

        x_total = np.stack(x_total)
        y_total = np.stack(y_total)

        whichLabelAbnormal = config["which_label_abnormal"]
        y_total = np.where(y_total == whichLabelAbnormal, 1, 0)

        return x_total, y_total


def touch_normal(dataSet, config, mode):

    assert mode in ["normal_only", "all"]

    if mode == "normal_only":

        normal_label = config["normal_label"]

        x = []
        y = []

        for eachData in dataSet:

            if eachData[1] == normal_label:
                x.append(eachData[0])
                y.append(0)

        x = np.stack(x)
        y = np.stack(y)

        return x, y

    else:

        normal_label = config["normal_label"]

        x = []
        y = []

        for eachData in dataSet:

            x.append(eachData[0])
            y.append(eachData[1])

        x = np.stack(x)
        y = np.stack(y)

        y = np.where(y == normal_label, 0, 1)

        return x, y


def change_data(dataSet, config, mode):

    if (
        config.get("normal_label") is not None
        and config.get("which_label_abnormal") is not None
    ):

        raise Exception(
            "normal_label 과 which_label_abnormal 둘다 존재합니다. 이 중 하나는 None이어야 합니다."
        )

    elif (
        config.get("normal_label") is None
        and config.get("which_label_abnormal") is None
    ):

        raise Exception(
            "normal_label 과 which_label_abnormal 둘다 None입니다. 하나는 값이 존재해야 합니다."
        )

    elif (
        config.get("normal_label") is not None
        and config.get("which_label_abnormal") is None
    ):

        data_X, data_y = touch_normal(dataSet=dataSet, config=config, mode=mode)
        print("returning normal data only")
        return data_X, data_y

    elif (
        config.get("normal_label") is None
        and config.get("which_label_abnormal") is not None
    ):

        data_X, data_y = touch_abnormal(dataSet=dataSet, config=config, mode=mode)
        print("converting data into binary")
        return data_X, data_y


def check_and_normalize(data_x, config):

    if (
        config.get("normal_label") is not None
        and config.get("which_label_abnormal") is not None
    ):

        raise Exception(
            "normal_label 과 which_label_abnormal 둘다 존재합니다. 이 중 하나는 None이어야 합니다."
        )

    elif (
        config.get("normal_label") is None
        and config.get("which_label_abnormal") is None
    ):

        raise Exception(
            "normal_label 과 which_label_abnormal 둘다 None입니다. 하나는 값이 존재해야 합니다."
        )

    elif (
        config.get("normal_label") is not None
        and config.get("which_label_abnormal") is None
    ):

        if config["normalize"] is True:

            if config["data_type"] == "mnist":

                mean = config["mnist_mean"]
                std = config["mnist_std"]

            elif config["data_type"] == "cifar":

                mean = config["cifar_mean"]
                std = config["cifar_std"]

            # elif config['data_type'] in [f'mnist_{i}' for i in range(1,11)]+[[f'cifar_{i}' for i in range(1,11)]]:
            elif config["data_type"] in [f"mnist_{i}" for i in range(1, 11)] + [
                f"cifar_{i}" for i in range(1, 11)
            ]:

                print(config["data_type"])

                with open(
                    "/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/data_download_path/noisedData/configs/noised_data_one_class_normal.pickle",
                    "rb",
                ) as F:

                    meanStdDict = pickle.load(F)

                which_data = config["data_type"].split("_")[0]
                which_noise = config["data_type"].split("_")[1]
                which_class = str(config.get("normal_label"))

                mean = meanStdDict[f"{which_data}_train_{which_noise}_{which_class}"][
                    "mean"
                ]
                std = meanStdDict[f"{which_data}_train_{which_noise}_{which_class}"][
                    "std"
                ]

                print(mean, std)

            return (data_x - mean) / std

        else:

            return data_x

    elif (
        config.get("normal_label") is None
        and config.get("which_label_abnormal") is not None
    ):

        if config["normalize"] is True:

            if config["data_type"] == "mnist":

                mean = config["mnist_mean"]
                std = config["mnist_std"]

            elif config["data_type"] == "cifar":

                mean = config["cifar_mean"]
                std = config["cifar_std"]

            elif config["data_type"] in [f"mnist_{i}" for i in range(1, 11)] + [
                [f"cifar_{i}" for i in range(1, 11)]
            ]:

                with open(
                    "/home/asdflkj3123/mainDir/forUni/theDir1/DeepLearningFinal/newDataVer0/scripts/data_download_path/noisedData/configs/noised_data_one_class_abnormal.pickle",
                    "rb",
                ) as F:

                    meanStdDict = pickle.load(F)

                which_data = config["data_type"].split("_")[0]
                which_noise = config["data_type"].split("_")[1]
                which_class = str(config.get("which_label_abnormal"))

                mean = meanStdDict[f"{which_data}_train_{which_noise}_{which_class}"][
                    "mean"
                ]
                std = meanStdDict[f"{which_data}_train_{which_noise}_{which_class}"][
                    "std"
                ]

            return (data_x - mean) / std

        else:

            return data_x


def dataSetToTensor(dataSet):

    x = []
    y = []

    for eachData in dataSet:
        
        x.append(eachData[0])
        y.append(eachData[1])

    x = np.stack(x)
    y = np.stack(y)
    print('x shape is : ',x.shape,y.shape)

    return x, y

def dataSetToTensor_testbed(dataSet,isTrain):
    
    if isTrain==True:
    
        x_train = []
        y_train = []
        
        x_val = []
        y_val = []
        
        for eachData in dataSet:
            
            eachData_data = eachData[0]
            eachData_label = eachData[1]
            eachData_flg = eachData[2]
            
            
            if eachData_flg == 'train':
                x_train.append(eachData_data)
                y_train.append(eachData_label)
            
            elif eachData_flg == 'val':
                x_val.append(eachData_data)
                y_val.append(eachData_label)
                
        x_train= np.stack(x_train)
        y_train= np.stack(y_train)
        x_val = np.stack(x_val)
        y_val = np.stack(y_val)
                
        return x_train,x_val,y_train,y_val

    else:
        x_test = []
        y_test = []
        
        for eachData in dataSet:
            
            eachData_data = eachData[0]
            
            eachData_label = eachData[1]
            
            x_test.append(eachData_data)
            y_test.append(eachData_label)
            

        x_test = np.stack(x_test)
        y_test = np.stack(y_test)
        
        return x_test, y_test
        
        

def convert_label_binary(label_tensor, config):

    if (
        config.get("normal_label") is not None
        and config.get("which_label_abnormal") is not None
    ):

        raise Exception(
            "normal_label 과 which_label_abnormal 둘다 존재합니다. 이 중 하나는 None이어야 합니다."
        )

    elif (
        config.get("normal_label") is None
        and config.get("which_label_abnormal") is None
    ):

        raise Exception(
            "normal_label 과 which_label_abnormal 둘다 None입니다. 하나는 값이 존재해야 합니다."
        )

    elif (
        config.get("normal_label") is not None
        and config.get("which_label_abnormal") is None
    ):

        normal_label = config["normal_label"]
        label_tensor = np.where(label_tensor == normal_label, 0, 1)
        print("touching normal label..")
        return label_tensor

    elif (
        config.get("normal_label") is None
        and config.get("which_label_abnormal") is not None
    ):

        abnormal_label = config["which_label_abnormal"]
        label_tensor = np.where(label_tensor == abnormal_label, 1, 0)
        print("converting data into binary")
        return label_tensor


def return_normal_only(x_train, y_train):

    filtered_x = []
    filtered_y = []

    for each_x, each_y in zip(x_train, y_train):

        if each_y == 0:

            filtered_x.append(each_x)
            filtered_y.append(each_y)

    filtered_x = np.stack(filtered_x)
    filtered_y = np.stack(filtered_y)
    
    return filtered_x, filtered_y
