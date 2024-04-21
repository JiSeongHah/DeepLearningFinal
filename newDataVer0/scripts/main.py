import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from Dataload import myNewDataset
from shutil import copytree, rmtree

print(sys.path)
from pprint import pprint
import yaml
from mainloop import MainLoop
from datetime import datetime


class runLoop:

    def __init__(self, configPath, savePath) -> None:

        self.configPath = configPath

        self.loadedConfig = self.readConfig(self.configPath)

        now = datetime.now()
        self.savePath = os.path.join(savePath,self.loadedConfig['which_model']+'_', str(round(now.timestamp())))
        os.makedirs(self.savePath)
        
        self.loadedConfig["modelSavePath"] = self.savePath

    def copySaveResult(resultPath, savePath):

        try:
            copytree(resultPath, savePath)
            print("saving old hisotry complete")

        except:
            raise Exception

    def removeCurrentResult(resultPath):

        try:
            print("removing current results ...")
            rmtree(resultPath)
            print("removing current result complete!!!")

        except:
            raise Exception

    def readConfig(self, yaml_path):
        print("loading config...")
        with open(yaml_path) as f:
            loadedConfig = yaml.load(f, Loader=yaml.FullLoader)

        print("loading config complete")
        print("loaded config is :")
        pprint(loadedConfig)

        return loadedConfig

    def saveConfig(self, yaml_path, configs):
        print("saving configs...")

        with open(os.path.join(yaml_path, "resultConfig.yaml"), "w") as f:
            yaml.dump(configs, f)

        print("saving configs complete!!")

    def run(self, doTestOnly):

        MAINLOOP = MainLoop(config=self.loadedConfig, doTestOnly=doTestOnly)

        resultConfig = MAINLOOP.runMainLoop()

        configSavePath = os.path.join(self.savePath, "configs/")
        os.makedirs(configSavePath)
        self.saveConfig(yaml_path=configSavePath, configs=resultConfig)

        print("all complete!!!")


if __name__ == "__main__":

    for i in range(10):

        configPath = "../configs/config.yaml"

        savePath = "./history/"

        MAIN = runLoop(configPath, savePath=savePath)

        MAIN.run(doTestOnly=False)
