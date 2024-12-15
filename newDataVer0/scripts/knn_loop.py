import os
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
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

class KnnLoop:
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
        
        # do nothing 
        saveDict = {}

        return saveDict

    def runTest(self, dataSet, loadedModel):
        
        FEed_testBed_dataLst = [f'FEed_800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'FEed_800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]
        
        if self.config['data_type'] in [f'800_0_noiseRatio_{round(0.1*i,1)}' for i in range(10)]+[f'800_45_noiseRatio_{round(0.1*i,1)}' for i in range(10)]:
            
            raise Exception('legacy error')
                        
        elif self.config['data_type'] in FEed_testBed_dataLst:
            
            raise Exception('legacy error')
                
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
        
        
        if self.config['data_type'] in ['cifar_1','cifar_2','mnist_1','mnist_2','cifar_yes','mnist_yes']:
            
            with open('./data_download_path/label_idx_arr_per_ratio_image_ver.pkl','rb') as F:
                
                filter_arr_dict = pickle.load(F)
            
            
            target_filter_arr_dict = filter_arr_dict[
                os.path.join(
                    [
                        self.config['data_type'].split('_')[0],
                        f'normal_label_{self.config['normal_label']}'
                    ]
                )
            ]
            
        testResult = {}
        for filter_k,filter_v in target_filter_arr_dict.items():
            
            
            
            filtered_x = x_test[filter_v]
            filtered_y = y_test[filter_v]
        
            print("knn model loading..")
            self.knnModel = NearestNeighbors(n_neighbors = self.neiNum)
            print("loading knn  model complete")
            print("knn test start...")
            
            self.knnModel.fit(filtered_x)
            
            dist,indices = self.knnModel.kneighbors(filtered_x)
            
            y_anomaly_score = np.mean(dist,axis=1)

            each_testResult = {
                "y_anomaly_score": y_anomaly_score,
                "y_test": filtered_y,
                "x_test": filtered_x,
            }
            
            testResult[filter_k] = each_testResult

            print("knn test complete!!!")
        
        return testResult

    def saveModel(self, trainResult):

        modelSavePath = self.config["modelSavePath"]

        print(f"saving trained model complete")

    def load_model(self):

        model_load_path = self.config["modelSavePath"]
        
        return {'do':'nothing'}
