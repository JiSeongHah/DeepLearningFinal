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
import copy


class runLoop:

    def __init__(self, configPath, savePath) -> None:

        self.configPath = configPath

        lst_loadedConfig = self.readConfig(self.configPath)

        for data_type in lst_loadedConfig["data_type_lst"]:
            for which_model in lst_loadedConfig["which_model_lst"]:
                for normal_label in lst_loadedConfig["normal_label_lst"]:
                    # for noise_ratio in lst_loadedConfig["noise_ratio_lst"]:
                    self.loadedConfig = copy.deepcopy(lst_loadedConfig)

                    self.loadedConfig["data_type"] = data_type
                    self.loadedConfig["which_model"] = which_model
                    self.loadedConfig["normal_label"] = normal_label
                    # self.loadedConfig["noise_ratio"] = noise_ratio

                    now = datetime.now()
                    self.savePath = os.path.join(
                        savePath,
                        self.loadedConfig["which_model"] + "_image_non_noise_new_start",
                        # self.loadedConfig["which_model"] + "_grad_test_run_test",
                        self.loadedConfig["data_type"],
                        "normal_label_" + str(self.loadedConfig["normal_label"]),
                        # f"noise_{noise_ratio}",
                        str(round(now.timestamp())),
                    )
                    # self.savePath = os.path.join(savePath,self.loadedConfig['which_model']+'_grad_test',self.loadedConfig['data_type'],'normal_label_'+str(self.loadedConfig['normal_label']), str(round(now.timestamp())))
                    os.makedirs(self.savePath)

                    self.loadedConfig["modelSavePath"] = self.savePath

                    self.run(doTestOnly=False)

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

    for i in range(2):

        configPath = "../configs/config.yaml"

        savePath = "./history/"

        MAIN = runLoop(configPath, savePath=savePath)

        # MAIN.run(doTestOnly=False)
