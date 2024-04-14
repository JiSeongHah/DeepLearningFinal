import torch
import numpy as np
from sklearn.model_selection import train_test_split

def convert_label(dataSet,config):

    x_train_total = []
    y_train_total = []

    for eachData in dataSet:
        x_train_total.append(eachData[0])
        y_train_total.append(eachData[1])

    x_train_total = np.stack(x_train_total)
    y_train_total = np.stack(y_train_total)

    whichLabelAbnormal = config["which_label_abnormal"]
    y_train_total = np.where(y_train_total == whichLabelAbnormal, 1, 0)
    
    return x_train_total, y_train_total

def get_normal_only(dataSet,config):
    
    normal_label = config['normal_label']
    
    x = []
    y = []
    
    for eachData in dataSet:
        if y == normal_label:
            x.append(eachData[0])
            y.append(eachData[1])
        
    x = np.stack(x)
    y= np.stack(y)
    
    return x, y
    
    
    
    return x_train, x_val, y_train, y_val

def change_data(dataSet,config):
    
    if config.get('normal_label') is not None and config.get('which_label_abnormal') is not None:
            
        raise Exception('normal_label 과 which_label_abnormal 둘다 존재합니다. 이 중 하나는 None이어야 합니다.')
        
    elif config.get('normal_label') is None and config.get('which_label_abnormal') is None:
        
        raise Exception('normal_label 과 which_label_abnormal 둘다 None입니다. 하나는 값이 존재해야 합니다.')
        
    elif config.get('normal_label') is not None and config.get('which_label_abnormal') is None:
        
        data_X,data_y = get_normal_only(dataSet=dataSet,config=config)
        print('returning normal data only')
        return data_X, data_y
        
    elif config.get('normal_label') is None and config.get('which_label_abnormal') is not None:
        
        data_X, data_y = convert_label(dataSet=dataSet,config=config)
        print('converting data into binary')
        return data_X, data_y

def check_and_normalize(data_x,config,mode):
    
    assert mode in ['train','test']
    
    if config['normalize'] is True:
        
        if config['data_type'] == 'mnist':
            
            mean = config['mnist_mean']
            std = config['mnist_std']
            
        elif config['data_type'] == 'mnist':
            
            mean = config['cifar_mean']
            std = config['cifar_std']
        
        return (data_x -mean)/std

    else:
        
        return data_x
    
    
    
    


