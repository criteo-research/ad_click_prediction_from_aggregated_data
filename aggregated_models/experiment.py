import os.path
import os
from aggregated_models.noiseDistributions import *
from aggregated_models.AggMRFModelWithAggPreds import *
from aggregated_models.validation import MetricsComputer, LLH, NLlh
from sklearn.metrics import roc_auc_score
from aggregated_models.agg_mrf_model import AggMRFModel, AggMRFModelParams
from aggregated_models.aggdataset import AggDataset
import numpy as np
from pathlib import Path
import pandas as pd
import gc


class Experiment:
    def __init__(self, name, ss=None):
        self.name = name
        self.path = "experiments/" + name + "/"
        self.configfile = self.path + "config.txt"
        self.logfile = self.path + "log.txt"
        self.ss = ss
        if not os.path.exists("experiments/"):
            os.mkdir("experiments/")
        if not os.path.exists(self.path):
            os.mkdir(self.path)
            self.totalNbIters = 0
        if os.path.exists(self.configfile):
            print("config found. loading model")
            self.load()

    def printAndLog(self, x):
        print(x)
        with open(self.logfile, "a+") as handle:
            handle.write(x + "\n")

    def defineModel(self, aggdataPath, buildModelStr, stepsize=0.001):  # a string with python code returning a model.

        configString = ""
        configString += f"self.aggdataPath = '{aggdataPath}'\n"
        configString += f"self.stepsize = {stepsize}\n"
        configString += f"self.buildModelStr = '''{buildModelStr}'''\n"

        if os.path.exists(self.configfile):
            previousConfig = self.getConfigStr()
            if previousConfig != configString:
                print("Config missmatch", previousConfig, configString)
                raise Exception("config already exists and differs from new one")
            else:
                print("config was already defined and identical")
                return self  # already loaded
        else:
            with open(self.configfile, "w") as handle:
                handle.write(configString)
        self.load()
        return self

    def getConfigStr(self):
        with open(self.configfile, "r") as file:
            configStr = file.read()
        return configStr

    def loadConfig(self):
        conf = self.getConfigStr()
        exec(conf)

    def loadAggdata(self):
        with open(self.aggdataPath, "rb") as handle:
            self.aggdata = AggDataset.load(handle)

    def load(self):
        self.loadConfig()
        # self.loadAggdata()
        self.loadModel()

    def resetSS(self, ss):
        self.ss = ss
        self.loadModel()

    def loadModel(self):
        ss = self.ss
        if self.totalNbIters > 0:
            self.model = AggMRFModel.load(self.path + "model", spark_session=ss)
            self.printAndLog(f"model succesfully reloaded at {self.totalNbIters} iterations. ")
        else:
            print("creating model")
            self.loadAggdata()
            aggdata = self.aggdata
            exec(self.buildModelStr + "\nself.model = model")
        self.label = self.model.label

    def savemodel(self):
        # self.model.save(self.path + "model_temp")  ## quick&dirty "recover from crash during save"
        self.model.save(self.path + "model")

    def validate(self, test):
        y = test[self.label].values
        p = self.model.predictDF(test, "p").p.values
        auc = roc_auc_score(y, p)
        nllh = NLlh(p, y)
        llh = LLH(p, y)
        nbiters = self.totalNbIters
        result = f" auc={auc:.3f},  nllh={nllh:.3f},  llh={llh:.4f},  nbiters={nbiters}"
        self.printAndLog(result)

    def run(self, test, nbiters=1000, logevery=10):
        self.printAndLog(f"starting to train for {nbiters} iters. logging every {logevery}")
        for i in range(0, int(nbiters / logevery)):
            self.model.fit(logevery, self.stepsize)
            self.savemodel()
            self.totalNbIters = self.totalNbIters + logevery
            if test is not None:
                self.validate(test)
            gc.collect()

    @property
    def totalNbIters(self):
        try:
            return np.load(self.path + "nbiters.npy")[0]
        except:
            print("cannot load nbiter.  first run ?")
            return 0

    @totalNbIters.setter
    def totalNbIters(self, value):
        np.save(self.path + "nbiters.npy", np.array([value]))
        self._nbiters = value
