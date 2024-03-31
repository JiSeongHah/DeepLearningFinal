import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Dataload import myNewDataset
from shutil import copytree, rmtree
print(sys.path)
from pprint import pprint
import yaml
    



class runLoop():
    
    def __init__(self,configPath) -> None:
        
        self.configPath = configPath
       
        loadedConfig = self.readConfig(self.configPath)
        
    def copySaveResult(resultPath,savePath):
        
        try:
            copytree(resultPath,savePath)
            print('saving old hisotry complete')
            
            
        except:
            raise Exception
        
    def removeCurrentResult(resultPath):
        
        try:
            print('removing current results ...')
            rmtree(resultPath)
            print('removing current result complete!!!')
            
        except:
            raise Exception
            
        
    def readConfig(yaml_path):
        print('loading config...')
        with open(yaml_path) as f:
            loadedConfig = yaml.load(f, Loader=yaml.FullLoader)
        
        print('loading config complete')
        print('loaded config is :')
        pprint(loadedConfig)    
        
        return loadedConfig
    
    def saveConfig(yaml_path,loadedConfig):
        print('saving configs...')
        
        with open(os.path.join(yaml_path,'config.yaml'), 'w') as f:
            yaml.dump(loadedConfig, f)
            
        print('saving configs complete!!')
            
    def run(self):
        
        
        
        
    
    
if __name__ == '__main__':
    
    configPath = '../configs/config.yaml'
    
    MAIN = runLoop(configPath)
    
    
    