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


class SimpleSampleSet:
    def __init__(
        self,
        nbSamples=None,
        nbfeatures=None,
        projections=None,
        rows=None,
    ):
        if rows is not None:
            self.rows = rows
        else:
            if projections is None:
                self.columns = np.zeros([nbfeatures, nbSamples], np.int32)
            else:
                if nbfeatures is not None and nbfeatures != len(projections):
                    raise ValueError("Inconsistent parameters for SimpleSampleSet")
                self.columns = SimpleSampleSet.sampleIndependent(projections, nbSamples)

    @property
    def Size(self):
        return self.columns.shape[1]

    def Df(self, featurenames):
        return pd.DataFrame(self.columns, featurenames).transpose()

    def set_data_from_rows(self, rows):
        self.rows = rows

    def get_rows(self):
        return self.rows

    @property
    def rows(self):
        return self.columns.transpose()

    @rows.setter
    def rows(self, value):
        self.columns = value.transpose()

    @property
    def data(self):
        return self.columns

    @data.setter
    def data(self, value):
        self.columns = value

    @staticmethod
    def sampleOneColumn(projection, nbSamples):
        counts = projection.Data
        probas = counts / sum(counts)
        cumprobas = np.cumsum(probas)
        rvalues = np.random.random_sample(nbSamples)
        varvalues = np.array([bisect.bisect(cumprobas, r) for r in rvalues])
        return varvalues

    @staticmethod
    def sampleIndependent(projections, nbSamples):
        return np.array([SimpleSampleSet.sampleOneColumn(p, nbSamples) for p in projections])


class SampleSet(SimpleSampleSet):
    def __init__(
        self,
        projections,
        nbSamples,
        model,
        sampleFromPY0=False,
        maxNbRowsPerSlice=50,
        rows=None,
    ):
        super().__init__(nbSamples, projections=projections, rows=rows)

        self.sampleFromPY0 = sampleFromPY0
        self.features = [p.feature for p in projections]
        self.featurenames = [f.Name for f in self.features]

        self.Update(model)

    def UpdateSampleWithGibbs(self, model, toto=0):
        self.columns = model.RunParallelGibbsSampler(self, maxNbRows=model.maxNbRowsPerSlice)
        self.Update(model)

    def Update(self, model):
        self._computedotprods(model)
        self._setweights()
        self._compute_prediction(model)

    def _computedotprods(self, model):
        lambdas = model.dotproducts(model.clickWeights, self.columns) + model.lambdaIntercept
        mus = model.dotproducts(model.displayWeights, self.columns) + model.muIntercept
        self.expmu = np.exp(mus)
        self.explambda = np.exp(lambdas)

    @property
    def pclick(self):
        return self.explambda / (1 + self.explambda)

    def _setweights(self):
        self.weights = np.ones(len(self.explambda))
        if self.sampleFromPY0:
            # Importance weights to compute expectations on P(X=x)  from samples of P(X=x |Y=0)
            #  with:
            #  P(X=x) := exp( K(x).mu ) ( 1+ exp( K(x).lambda )  ) / Z
            #  P(X=x | Y=0) := exp( K(x).mu ) / Z0
            #  Thus w(x) = ( 1+ exp( K(x).lambda )  ) Z0 / Z
            # We estimate Z0/Z  by using E(x) = 1  (ie self normalised IW)
            self.weights = 1 + self.explambda
        self.weights = self.weights / self.weights.sum()

    def _compute_prediction(self, model):
        self.prediction = model.parameters * 0
        for w in model.displayWeights.values():
            self.prediction[w.indices] = w.feature.Project_(self.columns, self.weights)  # Correct for grads
        for w in model.clickWeights.values():
            self.prediction[w.indices] = w.feature.Project_(self.columns, self.weights * self.pclick)

    def PredictInternal(self, model):
        self._computedotprods(model)
        self._compute_prediction(model)

    def GetPrediction(self, model):
        return self.prediction

    def sampleY(self):
        r = np.random.rand(len(self.explambda))
        self.y = 1 * (r < self.pclick)


# Exostive list of all possible samples 'x', used internally by AggMRFModel for exact computations
class FullSampleSet(SampleSet):
    def __init__(self, projections, model, maxNbRowsPerSlice=50):
        self.features = [p.feature for p in projections]
        self.featurenames = [f.Name for f in self.features]
        self.allcrossmods = False
        df = FullSampleSet.buildCrossmodsSample(projections)
        x = df[self.featurenames].values.transpose()

        super().__init__(
            projections,
            nbSamples=None,
            model=model,
            sampleFromPY0=False,
            maxNbRowsPerSlice=maxNbRowsPerSlice,
            rows=x.transpose(),
        )

    @staticmethod
    def countCrossmods(projections):
        features = [p.feature for p in projections]
        nbCrossModalities = np.prod([f.Size for f in features])
        return nbCrossModalities

    @staticmethod
    def buildCrossmodsSample(projections):
        features = [p.feature for p in projections]
        nbCrossModalities = FullSampleSet.countCrossmods(projections)
        if nbCrossModalities > MAXMODALITIES:
            print("Too many crossmodalities", nbCrossModalities, MAXMODALITIES)
            raise ValueError("Too many crossmodalities")
        crossmodalitiesdf = pd.DataFrame([[0, 1]], columns=["c", "probaSample"])
        for f in features:
            n = f.Size - 1  # -1 because last modality is "missing"
            modalities = np.arange(0, n)
            modalitiesdf = pd.DataFrame({f.Name: modalities})
            crossmodalitiesdf = pd.merge(crossmodalitiesdf, modalitiesdf.assign(c=0), on="c")
        return crossmodalitiesdf

    def _setweights(self):
        self.weights = self.expmu * (1 + self.explambda)
        self.weights = self.weights / self.weights.sum()

    def UpdateSampleWithGibbs(self, model, toto=0):
        self.Update(model)
