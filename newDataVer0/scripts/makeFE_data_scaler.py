from joblib import dump, load
import pickle


def openPickle(loadPath):
    
    with open(loadPath, 'rb') as f:
        data = pickle.load(f)
        
        return data
    
def savePickle(savePath,data):
    with open(savePath, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
    print(f'saving data into {savePath} complete')

import numpy as np
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

def saveScaler(dataType,savePath,intervalNum,noiseRatio):
    
    assert dataType in ['800_0','800_45']
    
    trainPath = f'./800_Testbed/pkled_data/FEed_data/FEed_{dataType}_interval_{intervalNum}_noiseRatio_{noiseRatio}/trainVal_{dataType}_interval_{intervalNum}_noised_{noiseRatio}.pkl'
    
    trainData = openPickle(trainPath)
    
    newTrainData = []
    for eachTrainData in trainData:
        if eachTrainData[2] == 'train':
            newTrainData.append(
                eachTrainData[0]
            )
        
        
    newTrainData = np.stack(newTrainData)
    print(newTrainData.shape)
    scaler = StandardScaler()
    
    scaler.fit(newTrainData)
    
    dump(scaler,f'./800_Testbed/pkled_data/FEed_data/FEed_{dataType}_interval_{intervalNum}_noiseRatio_{noiseRatio}/trainVal_FEed_{dataType}_interval_{intervalNum}_noised_{noiseRatio}_scaler.joblib')
    
    print(f'./800_Testbed/pkled_data/FEed_data/FEed_{dataType}_interval_{intervalNum}_noiseRatio_{noiseRatio}/trainVal_{dataType}_interval_{intervalNum}_noised_{noiseRatio}_mean_std.joblib', 'done')
    
    
    
    
        
dataType_lst = ['800_0','800_45']
interval_lst = [128,256,512,1024,2048,4096]

# noiseRatio_lst = ['0.0']
noiseRatio_lst = [round(i*0.1,1) for i in range(1,10)]

for dataType in dataType_lst:
    for interval_num in interval_lst:
        for noiseRatio in noiseRatio_lst:
            saveScaler(
                dataType=dataType,
                savePath= f'./800_Testbed/pkled_data/noised/FEed_{dataType}_noiseRatio_{noiseRatio}',
                intervalNum=interval_num,
                noiseRatio=noiseRatio
            )
            
            
        