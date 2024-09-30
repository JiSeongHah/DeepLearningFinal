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

    def __init__(self, configPath, savePath_root) -> None:

        self.configPath = configPath

        lst_loadedConfig = self.readConfig(self.configPath)

        for data_type in lst_loadedConfig["data_type_lst"]:
            for which_model in lst_loadedConfig["which_model_lst"]:
                for normal_label in lst_loadedConfig["normal_label_lst"]:
                    for noise_ratio in lst_loadedConfig["noise_ratio_lst"]:
                        loadedConfig = copy.deepcopy(lst_loadedConfig)

                        loadedConfig["data_type"] = data_type
                        loadedConfig["which_model"] = which_model
                        loadedConfig["normal_label"] = normal_label
                        loadedConfig["noise_ratio"] = noise_ratio

                        savePath_upper = os.path.join(
                            savePath_root,
                            loadedConfig["which_model"] + "_testbed_zscore_20240818",
                            # loadedConfig["which_model"] + "_grad_test",
                            loadedConfig["data_type"],
                            "normal_label_" + str(loadedConfig["normal_label"]),
                            f"noise_{noise_ratio}"
                        )
                        
                        dir_lst = os.listdir(savePath_upper)
                        for each_saved_dir in dir_lst:
                            savePath = os.path.join(
                                savePath_upper,
                                each_saved_dir
                            )
                            loadedConfig["modelSavePath"] = savePath
                            self.run(
                                loaded_config=loadedConfig,
                                save_path= savePath,
                                doTestOnly=True
                            )

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

    def saveConfig(self, yaml_path, configs,doTestOnly):
        print("saving configs...")
        if doTestOnly:
            with open(os.path.join(yaml_path, "resultConfig_testOnly.yaml"), "w") as f:
                yaml.dump(configs, f)
        else:
            with open(os.path.join(yaml_path, "resultConfig.yaml"), "w") as f:
                yaml.dump(configs, f)

        print("saving configs complete!!")

    def run(self, loaded_config,save_path,  doTestOnly):

        MAINLOOP = MainLoop(config=loaded_config, doTestOnly=doTestOnly)

        resultConfig = MAINLOOP.runMainLoop()

        configSavePath = os.path.join(save_path, "configs/")
        
        self.saveConfig(yaml_path=configSavePath, configs=resultConfig,doTestOnly=doTestOnly)

        print("all complete!!!")


if __name__ == "__main__":

    configPath = "../configs/config.yaml"

    savePath = "./history/"

    MAIN = runLoop(configPath, savePath=savePath)

    # MAIN.run(doTestOnly=False)
