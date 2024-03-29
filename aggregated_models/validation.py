import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import random


def LLH(prediction, y):
    llh = np.log(prediction) * y + np.log(1 - prediction) * (1 - y)
    return sum(llh) / len(y)


def Entropy(y):
    py = sum(y > 0) / len(y)
    return py * np.log(py) + (1 - py) * np.log(1 - py)


# normalized loglikelihood, or "% of entropy explained by the model", also known as "llh_comp_vn"
def NLlh(prediction, y):
    if any(prediction <= 0) or any(prediction >= 1):
        return np.nan
    h = Entropy(y)
    llh = LLH(prediction, y)
    return (h - llh) / h


def MSE(prediction, y):
    errors = prediction - y
    se = sum(errors * errors)
    return se / len(errors)


# "Proportion of variance explained by the model" , aka "mse comp vn"
def NMSE(prediction, y):
    mse = MSE(prediction, y)
    variance = np.var(y)
    return (variance - mse) / variance


class MetricsComputer:
    def __init__(self, label, pred_col_name="prediction"):
        self.label = label
        self.pred_col_name = pred_col_name

    def run(self, model, df):
        y = df[self.label].values
        predictedDf = self.getPredictedDf(model, df)
        predictions = predictedDf[self.pred_col_name]
        nllh = NLlh(predictions, y)
        nmse = NMSE(predictions, y)
        # aggse = self.AggSe(predictedDf , model.features)
        # aggL1 = self.AggL1Error(predictedDf , model.features)
        return f"NLLH={nllh:.4f}, NMSE={nmse:.4f}  "

    def getLLH(self, model, df):
        y = df[self.label].values
        predictedDf = self.getPredictedDf(model, df)
        predictions = predictedDf[self.pred_col_name]
        nllh = NLlh(predictions, y)
        return nllh

    def plot(self, model, df, features):
        a = self.getPredictedDf(model, df)
        a = a[features + [self.label, self.pred_col_name]].groupby(features).sum().reset_index()
        plt.plot(a[self.label], a[self.pred_col_name], "x")

    def getPredictedDf(self, model, df):
        df = model.predictDF(df, self.pred_col_name)
        return df

    def AggL1Error(self, dfWithPrediction, features):
        label = self.label
        se = 0.0
        for var in features:
            agg = dfWithPrediction[[var, label, self.pred_col_name]].groupby(var).sum().reset_index()
            agg["se"] = np.abs(agg[label] - agg[self.pred_col_name])
            se += agg["se"].sum()
        return se / dfWithPrediction[label].sum()

    def AggSe(self, dfWithPrediction, features):
        label = self.label
        se = 0.0
        for var in features:
            agg = dfWithPrediction[[var, label, self.pred_col_name]].groupby(var).sum().reset_index()
            agg["se"] = (agg[label] - agg[self.pred_col_name]) * (agg[label] - agg[self.pred_col_name])
            se += agg["se"].sum()
        return se / dfWithPrediction[label].sum()
