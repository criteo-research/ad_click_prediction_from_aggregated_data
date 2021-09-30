import pandas as pd
import numpy as np
import random
import bisect
from collections import Counter
from aggregated_models.featuremappings import (
    CrossFeaturesMapping,
    SingleFeatureMapping,
)

MAXMODALITIES = 5e7


# set of samples of 'x' used internally by AggMRFModel
class SampleSet:
    def __init__(
        self,
        projections,
        nbSamples=None,
        decollapseGibbs=False,
        sampleFromPY0=False,
        maxNbRowsPerSlice=50,
        rows=None,
    ):
        self.projections = projections
        self.decollapseGibbs = decollapseGibbs
        self.sampleFromPY0 = sampleFromPY0
        self.features = [p.feature for p in projections]
        self.featurenames = [f.Name for f in self.features]
        self.allcrossmods = False
        if rows is not None:
            self.set_data_from_rows(rows)
        elif nbSamples is None:
            df = self.buildCrossmodsSample()
            self.columns = df[self.featurenames].values.transpose()
        else:
            self.columns = self.sampleIndepedent(nbSamples)

        self.Size = len(self.columns[0])
        self.probaIndep = self.computeProbaIndep()
        self.probaSamples = self.probaIndep
        self._setweights()

        self.expmu = None
        self.explambda = None
        self.prediction = None

    def set_data_from_rows(self, rows):
        self.columns = rows.transpose()
        self.Size = len(self.columns[0])
        self.probaIndep = self.computeProbaIndep()
        self.probaSamples = self.probaIndep
        self._setweights()

    def get_rows(self):
        return self.columns.transpose()

    def _setweights(self):
        if self.allcrossmods:
            self.weights = np.ones(self.Size)
        else:
            scaling = 1.0 / self.Size
            self.weights = scaling / self.probaSamples

    def _computeProbaSamples(self, muIntercept, lambdaIntercept):
        if self.sampleFromPY0:
            # n = np.exp( self.muIntercept ) * ( 1 + np.exp(self.lambdaIntercept) )
            n = np.exp(muIntercept)
            self.probaSamples = (self.expmu) / n
        else:
            n = np.exp(muIntercept) * (1 + np.exp(lambdaIntercept))
            self.probaSamples = (self.expmu + self.explambda) / n

    def _compute_enoclick_eclick(self, muIntercept, lambdaIntercept):
        if self.allcrossmods:
            # exact computation
            self.Z = self.expmu.sum() + self.explambda.sum()
            n = np.exp(muIntercept) * (1 + np.exp(lambdaIntercept))
            self.Enoclick = self.expmu / self.Z * n
            self.Eclick = self.explambda / self.Z * n

        elif self.decollapseGibbs:
            # sampling Y instead of taking the expectation. Yeah it looks silly.
            pclick = self.explambda / (self.expmu + self.explambda)
            r = np.random.rand(len(pclick))
            clicked = 1 * (r < pclick)
            self.Enoclick = (1 - clicked) * (self.expmu + self.explambda) * self.weights
            self.Eclick = clicked * (self.expmu + self.explambda) * self.weights

        else:  # normal case (Gibbs samples)
            self.Enoclick = self.expmu * self.weights
            self.Eclick = self.explambda * self.weights
            if self.sampleFromPY0:  # correct importance weigthing formula
                z0_on_z = 1 / np.mean((1 + self.explambda / self.expmu))  # = P(Y)
                self.Eclick *= z0_on_z * (1 + np.exp(lambdaIntercept))
                self.Enoclick *= z0_on_z * (1 + np.exp(lambdaIntercept))

    def PredictInternal(self, model):
        self._computedotprods(model)
        self._compute_enoclick_eclick(model.muIntercept, model.lambdaIntercept)
        self._compute_prediction(model)

    def _compute_prediction(self, model):
        predict = model.parameters * 0
        for w in model.displayWeights.values():
            predict[w.indices] = w.feature.Project_(self.columns, self.Eclick + self.Enoclick)  # Correct for grads
        for w in model.clickWeights.values():
            predict[w.indices] = w.feature.Project_(self.columns, self.Eclick)
        self.prediction = predict

    def GetPrediction(self, model):
        return self.prediction

    def UpdateSampleWithGibbs(self, model, toto=0):
        self.columns = model.RunParallelGibbsSampler(self, maxNbRows=model.maxNbRowsPerSlice)

    def UpdateSampleWeights(self, model):
        self._computedotprods(model)
        self._computeProbaSamples(model.muIntercept, model.lambdaIntercept)
        self._setweights()
        self._compute_enoclick_eclick(model.muIntercept, model.lambdaIntercept)
        self._compute_prediction(model)

    def _computedotprods(self, model):
        lambdas = model.dotproducts(model.clickWeights, self.columns) + model.lambdaIntercept
        mus = model.dotproducts(model.displayWeights, self.columns) + model.muIntercept
        expmu = np.exp(mus)
        explambda = np.exp(lambdas) * expmu
        self.expmu = expmu
        self.explambda = explambda

    def sampleY(self):
        pclick = self.explambda / (self.expmu + self.explambda)
        r = np.random.rand(len(pclick))
        self.y = 1 * (r < pclick)

    def Df(self):
        return pd.DataFrame(self.columns, self.featurenames).transpose()

    def countCrossmods(self):
        nbCrossModalities = np.prod([f.Size for f in self.features])
        return nbCrossModalities

    def buildCrossmodsSample(self):
        self.allcrossmods = True
        nbCrossModalities = self.countCrossmods()
        if nbCrossModalities > MAXMODALITIES:
            print("Too many crossmodalities", nbCrossModalities, MAXMODALITIES)
            return self.sampleIndepedent(MAXMODALITIES)
        # else:
        #    #  print( f"Building full set of {nbCrossModalities:.1E}  crossmodalities ")
        crossmodalitiesdf = pd.DataFrame([[0, 1]], columns=["c", "probaSample"])
        for f in self.features:
            n = f.Size - 1  # -1 because last modality is "missing"
            modalities = np.arange(0, n)
            modalitiesdf = pd.DataFrame({f.Name: modalities})
            crossmodalitiesdf = pd.merge(crossmodalitiesdf, modalitiesdf.assign(c=0), on="c")
        return crossmodalitiesdf

    def sampleIndepedent(self, nbSamples):
        self.allcrossmods = False
        a = []
        for p in self.projections:
            counts = p.Data
            probas = counts / sum(counts)
            cumprobas = np.cumsum(probas)
            rvalues = np.random.random_sample(nbSamples)
            varvalues = np.array([bisect.bisect(cumprobas, r) for r in rvalues])
            a.append(varvalues)
        return np.array(a)

    def computeProbaIndep(self):
        df = self.Df()
        df["probaSample"] = 1.0
        for p in self.projections:
            counts = p.Data
            df["probaSample"] *= counts[df[p.feature.Name].values] / sum(counts)
        return df.probaSample.values

    @property
    def data(self):
        return self.columns

    @data.setter
    def data(self, value):
        self.columns = value
