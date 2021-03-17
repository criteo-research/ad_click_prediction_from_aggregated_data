import pandas as pd
import numpy as np
import random
import bisect
from collections import Counter
from agg_models.featuremappings import (
    CrossFeaturesMapping,
    FeatureMapping,
    DataProjection,
)

MAXMODALITIES = 1e7

# set of samples of 'x' used internally by AggMRFModel
class SampleSet:
    def __init__(
        self,
        projections,
        nbSamples=None,
        decollapseGibbs=False,
        sampleFromPY0=False,
        maxNbRowsperGibbsUpdate=50,
    ):
        self.projections = projections
        self.decollapseGibbs = decollapseGibbs
        self.sampleFromPY0 = sampleFromPY0
        self.features = [p.feature for p in projections]
        self.featurenames = [f.Name for f in self.features]
        if nbSamples is None:
            df = self.buildCrossmodsSample()
            self.data = df[self.featurenames].values.transpose()
        else:
            self.data = self.sampleIndepedent(nbSamples)

        self.Size = len(self.data[0])

        self.probaIndep = self.computeProbaIndep()
        self.probaSamples = self.probaIndep
        self.setweights()
        self.expmu = None
        self.explambda = None
        self.use_spark_rdd = False

    def setweights(self):
        if self.allcrossmods:
            self.weights = np.ones(self.Size)
        else:
            scaling = 1.0 / self.Size
            self.weights = scaling / self.probaSamples

    def computeProbaSamples(self, muIntercept, lambdaIntercept):
        if self.sampleFromPY0:
            # n = np.exp( self.muIntercept ) * ( 1 + np.exp(self.lambdaIntercept) )
            n = np.exp(muIntercept)
            self.probaSamples = (self.expmu) / n
        else:
            n = np.exp(muIntercept) * (1 + np.exp(lambdaIntercept))
            self.probaSamples = (self.expmu + self.explambda) / n

    def applyreweighting(self, muIntercept, lambdaIntercept):
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

        else: # normal case (Gibbs samples)
            self.Enoclick = self.expmu * self.weights
            self.Eclick = self.explambda * self.weights
            if self.sampleFromPY0:  # correct importance weigthing formula
                z0_on_z = 1 / np.mean((1 + self.explambda / self.expmu))  # = P(Y)
                # print( "z0onz", z0_on_z)
                self.Eclick *= z0_on_z * (1 + np.exp(lambdaIntercept))
                self.Enoclick *= z0_on_z * (1 + np.exp(lambdaIntercept))

        self.pdisplays = self.Eclick + self.Enoclick
        
    def Predict(self, model):
        expmu, explambda = model.computedotprods(self)
        self.expmu = expmu
        self.explambda = explambda
        self.applyreweighting(model.muIntercept, model.lambdaIntercept)
        
    def UpdateSampleWithGibbs(self, model):
        self.data = model.RunParallelGibbsSampler(
                self, nbsteps=model.nbsteps, maxNbRows=model.maxNbRowsperGibbsUpdate
            )
        
    def UpdateSampleWeights(self, model):
        model.computedotprods(self)
        self.computeProbaSamples(model.muIntercept, model.lambdaIntercept)
        self.setweights()
        self.applyreweighting(model.muIntercept, model.lambdaIntercept)

    def sampleY(self):
        pclick = self.explambda / (self.expmu + self.explambda)
        r = np.random.rand(len(pclick))
        self.y = 1 * (r < pclick)

    def Df(self):
        return pd.DataFrame(self.data, self.featurenames).transpose()

    def countCrossmods(self):
        nbCrossModalities = np.prod([f.Size for f in self.features])
        return nbCrossModalities

    def buildCrossmodsSample(self):
        self.allcrossmods = True
        nbCrossModalities = self.countCrossmods()
        if nbCrossModalities > MAXMODALITIES:
            print(f"too many crossmodalities ({nbCrossModalities:.1E}) ")
            return self.sampleIndepedent(MAXMODALITIES)
        # else:
        #    print( f"Building full set of {nbCrossModalities:.1E}  crossmodalities ")
        crossmodalitiesdf = pd.DataFrame([[0, 1]], columns=["c", "probaSample"])
        for f in self.features:
            n = f.Size - 1  # -1 because last modality is "missing"
            modalities = np.arange(0, n)
            modalitiesdf = pd.DataFrame({f.Name: modalities})
            crossmodalitiesdf = pd.merge(
                crossmodalitiesdf, modalitiesdf.assign(c=0), on="c"
            )
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
