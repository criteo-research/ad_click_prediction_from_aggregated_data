import numpy as np
import pandas as pd
from aggregated_models import CrossFeaturesSet
from aggregated_models import Optimizers
from aggregated_models.baseaggmodel import BaseAggModel

from aggregated_models.noiseDistributions import expectedGaussianApprox
from aggregated_models.noiseDistributions import expectedGaussianKnowingDataPlusNoiseAndSampledDataExpect

# Logistic regression model from categorical features.
#  It is trained from:
#   - the aggregated labels
#   - the full event level dataset without labels.
# Note that those data are sufficient to fully train a logistic model.
# Tests show there is no difference in perfomances with other implentation of logistic regression.
# Side note:  this was implemented to begin with as a way to test BaseAggModel,
# it was finally faster than the other logistic regression I has, and so was used in most tests.


class AggLogistic(BaseAggModel):
    def __init__(
        self,
        aggdata,
        features,
        clicksCfs="*&*",
        regulL2=1.0,
        rescaling=True,
        noiseModelScaling=None,
        sigma=None,
        modelDataSamplingWithNoise=False,
    ):
        super().__init__(aggdata, features)
        self.regulL2 = regulL2
        self.clicksCfs = CrossFeaturesSet.parseCFNames(self.features, clicksCfs)
        self.setProjections()  # building all weights and projections now
        self.setWeights()
        self.initParameters()
        self.nbCoefs = sum([w.feature.Size for w in self.clickWeights.values()])
        self.Data = self.getAggDataVector(self.clickWeights, self.clickProjections)

        self.Data = np.maximum(self.Data, 0)
        self.rescaling = rescaling
        self.nbIters = 0

        if noiseModelScaling is not None and sigma is None:
            if "variance" in aggdata.aggregations:
                variancesProjections = {var: aggdata.aggregations["variance"][var] for var in self.fs}
                variance = self.getAggDataVector(self.clickWeights, variancesProjections)
                sigma = np.sqrt(variance)
                sigma[sigma == 0] = 0.01
            else:
                print("variance not found inaggdata.aggregations. Cannot compute sigma.  Missconfig ?")
        if sigma is not None and noiseModelScaling is None:
            noiseModelScaling = 1
        self.sigma = sigma
        self.noiseModelScaling = noiseModelScaling
        self.modelDataSamplingWithNoise = modelDataSamplingWithNoise

    def setProjections(self):
        self.fs = self.features + self.clicksCfs
        self.clickProjections = {var: self.aggdata.aggClicks[var] for var in self.fs}

    def setWeights(self):
        featuresAndCfs = self.features + self.clicksCfs
        self.clickWeights, offset = self.prepareWeights(featuresAndCfs)
        self.parameters = np.zeros(offset)

    def setDisplays(self, train):
        if type(train) is pd.DataFrame:
            self.samples = self.DfToX(train)
        elif type(train) is np.ndarray:
            self.samples = train
        else:
            raise Exception("unkown type in aggLogistic.setDiaplays", type(train))

    def initParameters(self):
        nbclicks = self.aggdata.Nbclicks
        nbdisplays = self.aggdata.Nbdisplays
        self.lambdaIntercept = np.log((nbclicks + 0.001) / (nbdisplays - nbclicks))

    def prepareFit(self, df):
        self.setDisplays(df)
        self.update()
        self.setRescalingRatio()

    def setRescalingRatio(self):
        self.nbsamples = self.samples.shape[1]
        self.globalRescaling = self.aggdata.Nbdisplays / self.nbsamples

        if self.rescaling:
            self.displaysProjections = {var: self.aggdata.aggDisplays[var] for var in self.fs}
            self.aggregatedDisplays = self.getAggDataVector(self.clickWeights, self.displaysProjections)
            self.aggregatedDisplays = np.maximum(self.aggregatedDisplays, 0)

            self.aggregatedDisplaysInSamples = self.project(np.ones(self.nbsamples))
            self.rescalingRatio = (self.aggregatedDisplaysInSamples + 1) / (self.aggregatedDisplays + 1)

            self.rescalingRatio[self.aggregatedDisplays == 0] = 0

    def predictDFinternal(self, df, pred_col_name: str):
        dotprods = self.dotproductsOnDF(self.clickWeights, df) + self.lambdaIntercept
        df[pred_col_name] = 1.0 / (1.0 + np.exp(-dotprods))
        return df

    def predictinternal(self):
        lambdas = self.dotproducts(self.clickWeights, self.samples)
        lambdas += self.lambdaIntercept
        self.pclick = 1.0 / (1.0 + np.exp(-lambdas))

    def update(self):
        self.predictinternal()

    def getPredictionsVector(self):
        return self.project(self.pclick)

    @property
    def pData(self):
        return self.project(self.pclick)

    def project(self, v):
        x = self.parameters * 0
        for w in self.clickWeights.values():
            x[w.indices] = w.feature.Project_(self.samples, v)
        return x * self.globalRescaling

    # Computing approx LLH, (not true LLH)
    def computeLoss(self, epsilon=1e-12):
        llh = self.computeLossNoRegul(epsilon)
        regul = (self.parameters * self.parameters).sum() * self.regulL2
        return (llh + regul) / self.nbCoefs

    #  "approx" loss.
    def computeLossNoRegul(self, epsilon=1e-12):
        preds = self.getPredictionsVector()
        llh = -(self.Data * np.log(preds + epsilon) - preds).sum()
        llh += (self.Data * np.log(self.Data + epsilon) - self.Data).sum()
        return (llh) / self.nbCoefs

    # grad of "exact" loss
    def computeGradient(self):  # grad of loss
        preds = self.getPredictionsVector()
        gradient = -self.Data + preds
        if self.rescaling:
            # gradient = -self.Data * np.minimum(1, self.rescalingRatio) + preds / np.maximum(1, self.rescalingRatio)

            c_corrected = self.Data * self.rescalingRatio

            if self.sigma is not None:
                if self.modelDataSamplingWithNoise:
                    expedctedNoise = expectedGaussianKnowingDataPlusNoiseAndSampledDataExpect(
                        c_corrected, preds, self.sigma, self.globalRescaling
                    )
                else:
                    expedctedNoise = expectedGaussianApprox(c_corrected, preds, self.sigma)
                c_corrected -= expedctedNoise * self.noiseModelScaling

            gradient = -c_corrected + preds

            gradient[self.rescalingRatio <= 0] = 0  # No gradient where data are crasy

        gradient += 2 * self.parameters * self.regulL2
        self.normgrad = sum(gradient * gradient)
        return gradient

    def computeInvHessianDiagAtOptimum(self):  # grad of loss
        return 1 / (self.regulL2 * 2 + self.Data * self.rescalingRatio)

    def computeInvHessianDiag(self, alpha=0.5):  # grad of loss
        preds = self.getPredictionsVector()
        preds = (1 - alpha) * self.Data * np.minimum(1, self.rescalingRatio) + alpha * preds / np.maximum(
            1, self.rescalingRatio
        )
        # preds = preds * alpha + self.Data * (1 - alpha)  # averaging with value at optimum
        return 1 / (self.regulL2 * 2 + preds)

    def fit(self, train, nbIter=50, alpha=None):
        if not hasattr(self, "samples"):
            self.prepareFit(train)
        if alpha is None:
            alpha = 0.5 / len(self.clickWeights)
        self.fitSimple(nbIter, alpha)

    def fitSimple(self, nbIter=100, alpha=0.01):
        def endIterCallback():
            self.nbIters += 1

        Optimizers.simpleGradientStep(
            self,
            nbiter=nbIter,
            alpha=alpha,
            endIterCallback=endIterCallback,
        )

    def computeExactLoss(self, df):
        p = self.predictDF(df, "pclick").pclick.values
        y = df[self.label].values
        avgy = sum(y) / len(y)
        llh = -sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        llh -= -sum(y * np.log(avgy) + (1.0 - y) * np.log(1.0 - avgy))
        regul = sum(self.parameters * self.parameters) * self.regulL2
        return (regul + llh) / self.nbCoefs

    def __repr__(self):
        return f"logitic({self.features},λ={self.regulL2:.1E})"
