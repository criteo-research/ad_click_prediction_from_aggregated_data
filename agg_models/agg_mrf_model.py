import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from enum import Enum
import random
import bisect
from IPython.display import clear_output, display
from collections import Counter
from agg_models.featuremappings import (
    CrossFeaturesMapping,
    FeatureMapping,
    DataProjection,
)
from agg_models.SampleSet import SampleSet
from agg_models.SampleRdd import SampleRdd
from agg_models import featuremappings
from joblib import Parallel, delayed

from agg_models.baseaggmodel import BaseAggModel
from agg_models import Optimizers

from operator import add
from numba import jit


class ConstantMRFParameters:
    def __init__(
            self,
            nbSamples,
            nbParameters,
            sampleFromPY0,
            explosedDisplayWeights,
            displayWeights,
            clickWeights,
            modalitiesByVarId,
            muIntercept,
            lambdaIntercept
        ):
        self.nbSamples = nbSamples
        self.nbParameters = nbParameters
        self.sampleFromPY0 = sampleFromPY0
        self.explosedDisplayWeights = explosedDisplayWeights
        self.displayWeights = displayWeights
        self.clickWeights = clickWeights
        self.modalitiesByVarId = modalitiesByVarId
        self.muIntercept = muIntercept
        self.lambdaIntercept = lambdaIntercept
        self.enoclick = (1 + np.exp(self.lambdaIntercept)) * np.exp(self.muIntercept) / self.nbSamples
            


class VariableMRFParameters:
    def __init__(
            self,
            parameters
        ):
        self.parameters = parameters


class AggMRFModel(BaseAggModel):
    def __init__(
        self,
        aggdata,
        features,
        priorDisplays=0.5,  # used for initialization of weights associated to  "unknown modalities"
        exactComputation=False,  #
        nbSamples=1e5,  # Nb internal Gibbs samples
        regulL2=1.0,  # regularization parameter on 'mu'
        regulL2Click=None,  # regularization parameter on the parameters of P(Y|X). by default, same value as regulL2
        displaysCfs="*&*",
        clicksCfs="*&*",
        activeFeatures=None,  # Obsolete (was used to incrementaly learn the model)
        noiseDistribution=None,  # Parameters of the noise on the aggregated data (if any)   ,
        sampleFromPY0=False,
        maxNbRowsperGibbsUpdate=50,
        sparkSession=None
    ):
        super().__init__(aggdata, features)
        self.priorDisplays = priorDisplays
        self.exactComputation = exactComputation
        self.nbSamples = int(nbSamples) if not exactComputation else None

        self.regulL2 = regulL2

        self.displaysCfs = featuremappings.parseCFNames(self.features, displaysCfs)
        self.clicksCfs = featuremappings.parseCFNames(self.features, clicksCfs)

        self.allFeatures = self.features
        self.activeFeatures = activeFeatures if activeFeatures is not None else features
        # batch Size for the Gibbs sampler. (too high => memory issues on large models)
        self.maxNbRowsperGibbsUpdate = maxNbRowsperGibbsUpdate
        self.noiseDistribution = noiseDistribution
        self.sampleFromPY0 = sampleFromPY0
        # Compute Monte Carlo by sampling Y  (no good reason to do that ? )
        self.decollapseGibbs = False
        self.RaoBlackwellization = False

        self.regulL2Click = regulL2Click
        if regulL2Click is None:
            self.regulL2Click = regulL2

        self.sparkSession = sparkSession
        # Preparing weights, parameters, samples ...
        self.prepareFit()

    def setProjections(self):
        clickFeatures = self.features + self.clicksCfs
        self.clickProjections = {var: self.aggdata.aggClicks[var] for var in clickFeatures}
        displayFeatures = self.features + self.displaysCfs
        self.displayProjections = {var: self.aggdata.aggDisplays[var] for var in displayFeatures}
        self.allClickProjections = self.clickProjections.copy()
        self.allDisplayProjections = self.displayProjections.copy()

    def setWeights(self):
        self.displayWeights, self.offsetClicks = self.prepareWeights(self.features + self.displaysCfs)
        self.clickWeights, offset = self.prepareWeights(self.features + self.clicksCfs, self.offsetClicks)
        self.parameters = np.zeros(offset)
        self.allClickWeights = self.clickWeights.copy()
        self.allDisplayWeights = self.displayWeights.copy()
        self.regulVector = np.zeros(offset) + self.regulL2
        self.regulVector[self.offsetClicks :] = self.regulL2Click

    def buildSamples(self):
        samples = SampleSet(
            [self.displayProjections[var] for var in self.features],
            self.nbSamples,
            self.decollapseGibbs,
            self.sampleFromPY0,
            self.maxNbRowsperGibbsUpdate,
        )

        return samples

    def setSamples(self):
        self.samples = self.buildSamples()

    def initParameters(self):
        v0 = self.features[0]
        self.normgrad = 1.0
        nbclicks = self.aggdata.Nbclicks
        nbdisplays = self.aggdata.Nbdisplays
        self.muIntercept = np.log(nbdisplays - nbclicks)
        self.lambdaIntercept = np.log(nbclicks) - self.muIntercept
        logNbDisplay = np.log(nbdisplays)
        for var in self.activeFeatures:
            weights = self.displayWeights[var]
            proj = self.displayProjections[var]
            self.parameters[weights.indices] = np.log(np.maximum(proj.Data, self.priorDisplays)) - logNbDisplay
            # init to log( P(v | C=0 ) instead ???
        (
            exportedDisplayWeights,
            exportedClickWeights,
            modalitiesByVarId,
            parameters
        ) = self.exportWeightsAll()
        if self.sparkSession:
            self.constantMRFParameters = self.sparkSession.sparkContext.broadcast(
                    ConstantMRFParameters(
                        self.nbSamples,
                        self.parameters.size,
                        self.sampleFromPY0,
                        exportedDisplayWeights,
                        self.displayWeights,
                        self.clickWeights,
                        modalitiesByVarId,
                        self.muIntercept,
                        self.lambdaIntercept
                    )
                )
            self.variableMRFParameters = self.sparkSession.sparkContext.broadcast(
                    VariableMRFParameters(parameters)
                )

    def prepareFit(self):
        self.setProjections()  # building all weights and projections now
        self.setWeights()
        self.setActiveFeatures(self.activeFeatures)  # keeping only those active at the beginning
        self.initParameters()
        if self.sparkSession:
            self.buildSamplesRdd()
        self.update()
        return

    def isActive(self, v):
        return all([x in self.activeFeatures for x in v.split("&")])

    def setActiveFeatures(self, activeFeatures):
        self.features = activeFeatures
        self.activeFeatures = activeFeatures
        self.displayProjections = {v: f for (v, f) in self.allDisplayProjections.items() if self.isActive(v)}
        self.displayWeights = {v: f for (v, f) in self.allDisplayWeights.items() if self.isActive(v)}
        self.clickProjections = {v: f for (v, f) in self.allClickProjections.items() if self.isActive(v)}
        self.clickWeights = {v: f for (v, f) in self.allClickWeights.items() if self.isActive(v)}
        self.nbCoefs = sum([w.feature.Size for w in self.displayWeights.values()]) + sum(
            [w.feature.Size for w in self.clickWeights.values()]
        )
        self.bestLoss = 9999999999999999.0
        self.setSamples()  # reseting data
        self.setAggDataVector()

    def setparameters(self, x):
        self.parameters = x
        if self.sparkSession:
            self.variableMRFParameters = self.sparkSession.sparkContext.broadcast(
                    VariableMRFParameters(self.parameters)
                )
        self.update()

    def predictDFinternal(self, df):
        # compute dot product on each line
        df["lambda"] = self.dotproductsOnDF(self.clickWeights, df) + self.lambdaIntercept
        df["mu"] = self.dotproductsOnDF(self.displayWeights, df) + self.muIntercept
        df["expmu"] = np.exp(df["mu"])
        df["explambda"] = np.exp(df["lambda"]) * df["expmu"]
        if "weight" in df.columns:
            df["E(NbNoClick)"] = df["expmu"] * df["weight"]
            df["E(NbClick)"] = df["explambda"] * df["weight"]
            df["E(NbDisplays)"] = df["E(NbClick)"] + df["E(NbNoClick)"]
        df["pclick"] = 1.0 / (1.0 + np.exp(-df["lambda"]))
        return df

    def computedotprods(self, samples):
        lambdas = self.dotproducts(self.clickWeights, samples.data) + self.lambdaIntercept
        mus = self.dotproducts(self.displayWeights, samples.data) + self.muIntercept
        expmu = np.exp(mus)
        explambda = np.exp(lambdas) * expmu
        samples.expmu = expmu
        samples.explambda = explambda

    def predictinternal(self, samples):
        if samples.use_spark_rdd:
            samples.Predict(self)
        else:

    def update(self):
        self.predictinternal(self.samples)

    def updateSamplesWithGibbs(self, samples):

        if not samples.allcrossmods:
            # Not applying Gibbs if full samples was generated
            samples.UpdateSampleWithGibbs(self)

        samples.UpdateSampleWeights(self)

    def compute_rdd_expdotproducts(self, rdd_samplesWithWeights):  
        
        constantMRFParameters = self.constantMRFParameters
        variableMRFParameters = self.variableMRFParameters

        def expdotproducts(x):
            t_x = x[0].transpose()
            mus = np.zeros(x[0].shape[0])
            lambdas = np.zeros(x[0].shape[0])
            for w in constantMRFParameters.value.displayWeights.values():
                mus += variableMRFParameters.value.parameters[w.feature.Values_(t_x) + w.offset]
            for w in constantMRFParameters.value.clickWeights.values():
                lambdas += variableMRFParameters.value.parameters[w.feature.Values_(t_x) + w.offset]
            mus = np.exp(mus + constantMRFParameters.value.muIntercept)
            return x[0], x[1], mus, mus * np.exp(lambdas + constantMRFParameters.value.lambdaIntercept)

        return rdd_samplesWithWeights.map(expdotproducts)

    def compute_enoclick_eclick(self, xweightsmulambdas):
        
        constantMRFParameters = self.constantMRFParameters

        def _computePDisplays(tuple_x_weights_exp_mu_lambda):
            x, w, expmu, explambda = tuple_x_weights_exp_mu_lambda
            enoclick = constantMRFParameters.value.enoclick
            lambda_mu = tuple_x_weights_exp_mu_lambda[3] / tuple_x_weights_exp_mu_lambda[2]
            z_i = (1 + lambda_mu).sum() / constantMRFParameters.value.nbSamples
            eclick = enoclick * lambda_mu
            return x, w, enoclick, eclick, z_i

        return xweightsmulambdas.map(_computePDisplays)

    def compute_enoclick_eclick_withweight(self, xweightsmulambdas):
        # x, expmu, explambda
        constantMRFParameters = self.constantMRFParameters
                
        def _computePDisplays(tuple_x_weight_mu_lambda):
            # x, expmu, explambda, weight
            weights = tuple_x_weight_mu_lambda[1]            
            expmu = tuple_x_weight_mu_lambda[2]
            explambda = tuple_x_weight_mu_lambda[3]
            enoclick = expmu * weights
            eclick = explambda * weights
            z_i = (1 + explambda /expmu).sum()
            if constantMRFParameters.value.sampleFromPY0:  # correct importance weigthing formula
                eclick *= (1 + np.exp(constantMRFParameters.value.lambdaIntercept))
                enoclick *= (1 + np.exp(constantMRFParameters.value.lambdaIntercept))
            return tuple_x_weight_mu_lambda[0], tuple_x_weight_mu_lambda[1], enoclick, eclick, z_i

        return xweightsmulambdas.map(_computePDisplays)

    def getPredictionsVector(self, samples):

        if samples.use_spark_rdd:
            return samples.predictions

        if self.RaoBlackwellization:
            return ComputeRWpred(self, self.samples, self.maxNbRowsperGibbsUpdate)

        x = self.parameters * 0
        for w in self.displayWeights.values():
            x[w.indices] = w.feature.Project_(samples.data, samples.pdisplays)  # Correct for grads
        for w in self.clickWeights.values():
            x[w.indices] = w.feature.Project_(samples.data, samples.Eclick)
        return x

    def getPredictionsVectorRdd(self, x_enoclick_eclick):
        
        constantMRFParameters = self.constantMRFParameters

        def computePredictions(tuple_x_enoclick_eclick):
            p = np.zeros(constantMRFParameters.value.nbParameters)
            t_x = tuple_x_enoclick_eclick[0].transpose()
            enoclick = tuple_x_enoclick_eclick[2]
            eclick = tuple_x_enoclick_eclick[3]
            zi = tuple_x_enoclick_eclick[4]
            for w in constantMRFParameters.value.displayWeights.values():
                p[w.indices] = w.feature.Project_(t_x, enoclick + eclick)  # Correct for grads
            for w in constantMRFParameters.value.clickWeights.values():
                p[w.indices] = w.feature.Project_(t_x, eclick)
            return p, zi

        return x_enoclick_eclick.map(computePredictions).reduce(lambda p_z,p_z1: (p_z[0]+p_z1[0], p_z[1]+p_z1[1]))

    def getPredictionsVector_(self, samples, index):
        x = self.parameters * 0
        for w in self.displayWeights.values():
            x[w.indices] = w.feature.Project_(samples.data[:, index], samples.pdisplays[index])  # Correct for grads
        for w in self.clickWeights.values():
            x[w.indices] = w.feature.Project_(samples.data[:, index], samples.Eclick[index])
        return x

    def setAggDataVector(self):
        self.Data = self.getAggDataVector(self.clickWeights, self.clickProjections)
        self.Data += self.getAggDataVector(self.displayWeights, self.displayProjections)
        self.DataRemoveNegatives = self.Data * (self.Data > 0)

    # Computing approx LLH, (not true LLH)
    def computeLoss(self, samples=None, epsilon=1e-12):
        llh = self.computeLossNoRegul(samples, epsilon)
        regul = (self.parameters * self.parameters * self.regulVector).sum()
        return llh + regul / self.nbCoefs

    #  "approx" loss.
    def computeLossNoRegul(self, samples=None, epsilon=1e-12):
        if samples is None:
            samples = self.samples
        preds = self.getPredictionsVector(samples)
        llh = -(self.DataRemoveNegatives * np.log(preds + epsilon) - preds).sum()
        llh += (self.DataRemoveNegatives * np.log(self.DataRemoveNegatives + epsilon) - self.DataRemoveNegatives).sum()
        return (llh) / self.nbCoefs

    # grad of "exact" loss
    def computeGradient(self):  # grad of loss
        return self.recomputeGradient(self.samples)

    def recomputeGradient(self, samples):  # grad of loss
        predictions = self.getPredictionsVector(samples)
        gradient = -self.Data + predictions
        if self.noiseDistribution is not None:
            noise = self.expectedNoise(predictions, samples)
            gradient += noise  # - (data-noise - preds)
        gradient += 2 * self.parameters * self.regulVector
        self.normgrad = sum(gradient * gradient)
        return gradient

    # Estimating E( L | D = data ,parmeters )  where L ~ Laplace and D = L + AggregatedCounts
    def expectedNoise(self, predictions, samples):
        # approximation: assuming that (Li) are independent knowing D.
        return self.expectedNoiseIndepApprox(predictions, samples)

    def expectedNoiseIndepApprox(self, predictions, samples=None):
        if predictions is None:
            predictions = self.getPredictionsVector(samples)
        return self.noiseDistribution.expectedNoiseApprox(self.Data, predictions)

    def sampleNoiseProba(self, samples):
        currentsamples = np.random.binomial(1, 0.5, samples.Size)
        index = np.where(currentsamples)[0]
        currentCounts = 2 * self.getPredictionsVector_(samples, index)
        currentNoise = self.Data - currentCounts

        logNoiseProba = self.noiseDistribution.LogProba(currentNoise).sum()
        # logNoiseProba = -np.abs(currentNoise).sum() *self.laplaceEpsilon
        return logNoiseProba

    #  LLH(data) . Only available if exactcomputations = True
    def ExactLoss(self, df):
        df = self.predictDF(df.copy())
        return np.mean(np.log((df["expmu"] * (1 - df.click) + df["explambda"] * df.click).values / self.getZ()))

    # todo:  estimation when exactcomputations != True
    def getZ(self):
        return self.samples.Z

    def ApproxLLHAggData(self, nbsamples, data=None, samples=None):
        if data is None:
            data = self.Data
        degrees = np.ones(len(self.parameters))
        degree_simplefeatures = 1 - (len(self.features) - 1)
        for f in self.features:
            degrees[self.displayWeights[f].indices] = degree_simplefeatures
            degrees[self.clickWeights[f].indices] = degree_simplefeatures
        Z = self.getZ()

        n = int(len(self.parameters) / 2)
        ind = np.arange(n)
        approxLLH = self.parameters.dot(data) / nbsamples
        nbclicks = data[self.clickWeights[self.features[0]].indices].sum()
        approxLLH += self.lambdaIntercept * nbclicks / nbsamples + self.muIntercept
        approxLLH -= np.log(Z)
        h = nbsamples * (np.log(nbsamples) - 1) - ((data * degrees).dot(np.log(data + 1) - 1)) / nbsamples
        # - (data*degrees) .dot(np.log(data+1)-1))/nbsamples
        return approxLLH

    def computeInvHessianDiagAtOptimum(self):  # grad of loss
        return 1 / (self.regulVector * 2 + self.DataRemoveNegatives)

    def computeInvHessianDiag(self, alpha=0.5):  # grad of loss
        preds = self.getPredictionsVector(self.samples)
        preds = preds * alpha + self.DataRemoveNegatives * (1 - alpha)  # averaging with value at optimum
        return 1 / (self.regulVector * 2 + preds)

    # Inv diagonal hessian , (avgerage berween current point and optimum)
    def computePrecondGradient(self, samples):  # grad of loss
        preds = self.getPredictionsVector(samples)
        gradient = -self.Data + preds
        gradient += 2 * self.parameters * self.regulVector
        precond = self.regulVector * 2 + (preds + self.DataRemoveNegatives) * 0.5
        return gradient / precond

    def updateAllSamplesWithGibbs(self):
        self.updateSamplesWithGibbs(self.samples)

    def buildSamplesRdd(self):        
        maxNbRows = self.maxNbRowsperGibbsUpdate
        rows = self.samples.data.transpose()
        weights = self.samples.weights.transpose()
        starts = np.arange(0, len(rows), maxNbRows)
        slices = [(rows[start: start + maxNbRows], weights[start: start + maxNbRows]) for start in starts]
        self.samples = SampleRdd(self.sparkSession.sparkContext.parallelize(slices))


    def updateSamplesWithGibbsRdd(self, samplesRdd):
        constantMRFParameters = self.constantMRFParameters
        variableMRFParameters = self.variableMRFParameters
        
        def myfun_sampling_from_p_y0(sampleWithWeight):
            sample = fastGibbsSampleFromPY0(
                constantMRFParameters.value.explosedDisplayWeights,
                constantMRFParameters.value.modalitiesByVarId,
                variableMRFParameters.value.parameters,
                sampleWithWeight[0],
                1,
            )
            return sample, sampleWithWeight[1]

        return samplesRdd.map(myfun_sampling_from_p_y0)

    def maxprobaratio(self, samples):
        probaSamples = samples.probaSamples
        pmodel = samples.expmu / np.exp(self.muIntercept)
        return max(pmodel / probaSamples)

    def fit(self, nbIter=100, alpha=0.01):
        lineSearch = Optimizers.followGradWithLinesearch(
            self,
            nbiter=nbIter,
            alpha=alpha,
            usePrecondAtOptim=False,
            usePrecondInDescent=True,
            endIterCallback=lambda: self.updateAllSamplesWithGibbs(),
        )
        return lineSearch.alpha

    def predictDF(self, df):
        df = self.transformDf(df)
        return self.predictDFinternal(df)

    # export data useful to compute dotproduct
    def exportWeights(self, weights):
        weightsByVar = {}
        for var in self.features:
            weightsByVar[var] = [x for x in weights.values() if var in x.feature.Name]
        allcoefsv = []
        allcoefsv2 = []
        alloffsets = []
        allotherfeatureid = []
        allmodulos = []

        for var in self.features:
            coefsv = []
            coefsv2 = []
            offsets = []
            otherfeatureid = []
            modulos = []
            for w in weightsByVar[var]:
                offsets += [w.offset]
                if type(w.feature) is featuremappings.CrossFeaturesMapping:
                    if var == w.feature._v1:
                        coefsv += [1]
                        coefsv2 += [w.feature.coefV2]
                        otherfeatureid += [self.features.index(w.feature._v2)]
                        modulos += [w.feature.Modulo]
                        # formula for indice of the crossmodality (V1=v1,V2=v2) in the parameter vector is :
                        #  w.offset + v1 * coefsv + v2 * coefsv2
                    else:
                        coefsv += [w.feature.coefV2]
                        coefsv2 += [1]
                        otherfeatureid += [self.features.index(w.feature._v1)]
                        modulos += [w.feature.Modulo]

                else:
                    coefsv += [1]
                    coefsv2 += [0]
                    otherfeatureid += [0]
                    modulos += [w.feature.Modulo]

            allcoefsv += [coefsv]
            allcoefsv2 += [coefsv2]
            alloffsets += [offsets]
            allotherfeatureid += [otherfeatureid]
            allmodulos += [modulos]
        return allcoefsv, allcoefsv2, alloffsets, allotherfeatureid, allmodulos

    def exportWeightsAll(self):
        exportedDisplayWeights = self.exportWeights(self.displayWeights)
        exportedClickWeights = self.exportWeights(self.clickWeights)
        # allcoefsv,allcoefsv2, alloffsets, allotherfeatureid = exportedDisplayWeights
        modalitiesByVarId = []
        for i in range(0, len(self.features)):
            var = self.features[i]
            modalitiesByVarId.append(np.arange(0, self.displayWeights[var].feature.Size - 1))
        modalitiesByVarId = (
            *(np.array(a) for a in modalitiesByVarId),
        )  # converting to tuple of np.array seems to make numba happier
        return (
            np.array(exportedDisplayWeights),
            np.array(exportedClickWeights),
            modalitiesByVarId,
            self.parameters,
        )

    def RunParallelGibbsSampler(self, samples, maxNbRows=1000):
        samples.sampleY()
        (
            exportedDisplayWeights,
            exportedClickWeights,
            modalitiesByVarId,
            parameters,
        ) = self.exportWeightsAll()
        start = 0

        rows = samples.data.transpose()

        starts = np.arange(0, len(rows), maxNbRows)
        slices = [(rows[start : start + maxNbRows], samples.y[start : start + maxNbRows]) for start in starts]

        def myfun(s):
            s = fastGibbsSample(
                exportedDisplayWeights,
                exportedClickWeights,
                modalitiesByVarId,
                parameters,
                s[0],
                self.nbGibbsSteps,
                s[1],
            )
            return s

        def myfun_sampling_from_p_y0(s):
            s = fastGibbsSampleFromPY0(exportedDisplayWeights, modalitiesByVarId, parameters, s[0], self.nbGibbsSteps)
            return s

        if samples.sampleFromPY0:
            myfun = myfun_sampling_from_p_y0

        runner = Parallel(n_jobs=14)
        jobs = [delayed(myfun)(myslice) for myslice in slices]
        samplesSlices = runner(jobs)
        # samplesSlices = [ myfun(myslice) for myslice in  slices ]
        return np.vstack(samplesSlices).transpose()


@jit(nopython=True)
def mybisect(a: np.ndarray, x):
    """Similar to bisect.bisect() or bisect.bisect_right(), from the built-in library."""
    n = a.size
    if n == 1:
        return 0
    left = 0
    right = n
    while right > left + 1:
        current = int((left + right) / 2)
        if x >= a[current]:
            left = current
        else:
            right = current
    return left


@jit(nopython=True)
def weightedSampleNUMBA(p):
    cp = np.cumsum(p)
    r = np.random.random() * cp[len(cp) - 1]
    return mybisect(cp, r)


def weightedSample(p):
    cp = np.cumsum(p)
    r = np.random.random() * cp[len(cp) - 1]
    return bisect.bisect(cp, r)


@jit(nopython=True)
def weightedSamplesNUMBA(M):
    return np.array([weightedSampleNUMBA(p) for p in M])


def weightedSamples(M):
    return np.array([weightedSample(p) for p in M])


# @jit(nopython=True) # Not working, cannot remember why.
def fastGibbsSample(
    exportedDisplayWeights,
    exportedClickWeights,
    modalitiesByVarId,
    paramsVector,
    x,
    nbsteps,
    y,
):
    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights
    (
        click_allcoefsv,
        click_allcoefsv2,
        click_alloffsets,
        click_allotherfeatureid,
        click_allmodulos,
    ) = exportedClickWeights

    x = x.transpose().copy()
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    nbsamples = x.shape[1]

    # Iterating on nbsteps steps
    for i in np.arange(0, nbsteps):

        # List of features ( one feature <=> its index in the arrays )
        f = np.arange(0, x.shape[0])
        #  Gibbs sampling may converge faster if the order in which we sample features is randomized.
        np.random.shuffle(f)

        # For each feature, ressample this feature conditionally to the other
        for varId in f:

            # data describing the different crossfeatures involving varId  in " K(x).mu"
            # Those things are arrays, of len the number of crossfeatures involving varId.
            disp_coefsv = allcoefsv[varId]
            disp_coefsv2 = allcoefsv2[varId]
            disp_offsets = alloffsets[varId]
            disp_otherfeatureid = allotherfeatureid[varId]
            disp_modulos = allmodulos[varId]
            # idem, but crossfeatures in " K(x).theta" part of the model
            click_coefsv = click_allcoefsv[varId]
            click_coefsv2 = click_allcoefsv2[varId]
            click_offsets = click_alloffsets[varId]
            click_otherfeatureid = click_allotherfeatureid[varId]
            click_modulos = click_allmodulos[varId]

            # array of possible modalities of varId
            modalities = modalitiesByVarId[varId]  # Should be 0,1,2 ... NbModalities
            # for each modality, compute P( modality | other features ) as exp( dotproduct)
            # initializing dotproduct
            mus = np.zeros((nbsamples, len(modalities)))
            lambdas = np.zeros((nbsamples, len(modalities)))

            # Computing the dotproducts
            #  For each crossfeature containing varId
            for varJ in np.arange(0, len(disp_coefsv)):
                modulo = disp_modulos[varJ]

                # let m a modality of feature varId, and m' a modality of the other feature
                #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
                # values of m' in the data
                modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
                # all possible modality m
                mods = modalities * disp_coefsv[varJ]
                # Computing crossfeatures
                # this is a matrix of shape (nbSamples, nbModalities of varId)
                crossmods = (np.add.outer(modsJ, mods) % modulo) + disp_offsets[varJ]
                # Adding crossfeature weight.
                mus += paramsVector[crossmods]

            for varJ in np.arange(0, len(click_coefsv)):
                modulo = click_modulos[varJ]

                modsJ = x[click_otherfeatureid[varJ]] * click_coefsv2[varJ]
                mods = modalities * click_coefsv[varJ]
                crossmods = (np.add.outer(modsJ, mods) % modulo) + click_offsets[varJ]
                # mus += paramsVector[crossmods] # buggycode

                lambdas += paramsVector[crossmods]

            mus = np.exp(mus)
            lambdas = np.exp(lambdas)

            mus = mus * ((lambdas.transpose() * y + 1 - y)).transpose()
            # mus = mus * ( 1  +  lambdas) # buggycode
            # Sampling now modality of varId
            varnew = weightedSamples(mus)
            # updating the samples
            x[varId] = varnew

    return x.transpose()


# Sampling X from P( X | Y = 0, mu)
# - Should be twice faster than sampling from P(X,Y) (no need to compute P(Y|X) part of the model)
# - We can later use importance weighting to correct for this (and not so different anyway)


def fastGibbsSampleFromPY0(exportedDisplayWeights, modalitiesByVarId, paramsVector, x, nbsteps):
    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights

    x = x.transpose().copy()
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    nbsamples = x.shape[1]

    # Iterating on nbsteps steps
    for i in np.arange(0, nbsteps):

        # List of features ( one feature <=> its index in the arrays )
        f = np.arange(0, x.shape[0])
        #  Gibbs sampling may converge faster if the order in which we sample features is randomized.
        np.random.shuffle(f)

        # For each feature, ressample this feature conditionally to the other
        for varId in f:

            # data describing the different crossfeatures involving varId  in " K(x).mu"
            # Those things are arrays, of len the number of crossfeatures involving varId.
            disp_coefsv = allcoefsv[varId]
            disp_coefsv2 = allcoefsv2[varId]
            disp_offsets = alloffsets[varId]
            disp_otherfeatureid = allotherfeatureid[varId]
            disp_modulos = allmodulos[varId]

            # array of possible modalities of varId
            modalities = modalitiesByVarId[varId]  # Should be 0,1,2 ... NbModalities
            # for each modality, compute P( modality | other features ) as exp( dotproduct)
            # initializing dotproduct
            mus = np.zeros((nbsamples, len(modalities)))

            # Computing the dotproducts
            #  For each crossfeature containing varId
            for varJ in np.arange(0, len(disp_coefsv)):
                modulo = disp_modulos[varJ]
                # let m a modality of feature varId, and m' a modality of the other feature
                #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
                # values of m' in the data
                modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
                # all possible modality m
                mods = modalities * disp_coefsv[varJ]
                # Computing crossfeatures
                # this is a matrix of shape (nbSamples, nbModalities of varId)
                crossmods = (np.add.outer(modsJ, mods) % modulo) + disp_offsets[varJ]
                # Adding crossfeature weight.
                mus += paramsVector[crossmods]

            mus = np.exp(mus)
            # Sampling now modality of varId
            varnew = weightedSamples(mus)
            # updating the samples
            x[varId] = varnew
    return x.transpose()


def cos(g, g2):
    return g.dot(g2) / np.sqrt(g.dot(g) * g2.dot(g2))


# @jit(nopython=True)
def computeRaoBlackwellisedExpectations(
    exportedDisplayWeights, exportedClickWeights, modalitiesByVarId, paramsVector, x, py
):

    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights
    (
        click_allcoefsv,
        click_allcoefsv2,
        click_alloffsets,
        click_allotherfeatureid,
        click_allmodulos,
    ) = exportedClickWeights

    x = x.transpose()
    nbsamples = x.shape[1]

    results = np.zeros(len(paramsVector))

    # List of features ( one feature <=> its index in the arrays )
    f = np.arange(0, x.shape[0])
    #  Gibbs sampling may converge faster if the order in which we sample features is randomized.

    # For each feature, ressample this feature conditionally to the other
    for varId in f:

        # data describing the different crossfeatures involving varId  in " K(x).mu"
        # Those things are arrays, of len the number of crossfeatures involving varId.
        disp_coefsv = allcoefsv[varId]
        disp_coefsv2 = allcoefsv2[varId]
        disp_offsets = alloffsets[varId]
        disp_otherfeatureid = allotherfeatureid[varId]
        disp_modulos = allmodulos[varId]

        # idem, but crossfeatures in " K(x).theta" part of the model
        click_coefsv = click_allcoefsv[varId]
        click_coefsv2 = click_allcoefsv2[varId]
        click_offsets = click_alloffsets[varId]
        click_otherfeatureid = click_allotherfeatureid[varId]
        click_modulos = click_allmodulos[varId]

        # array of possible modalities of varId
        modalities = modalitiesByVarId[varId]  # Should be 0,1,2 ... NbModalities
        # for each modality, compute P( modality | other features ) as exp( dotproduct)
        # initializing dotproduct
        mus = np.zeros((nbsamples, len(modalities)))
        lambdas = np.zeros((nbsamples, len(modalities)))

        # Computing the dotproducts
        #  For each crossfeature containing varId
        for varJ in np.arange(0, len(disp_coefsv)):
            modulo = disp_modulos[varJ]

            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
            # all possible modality m
            mods = modalities * disp_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)
            crossmods = (np.add.outer(modsJ, mods) % modulo) + disp_offsets[varJ]
            # Adding crossfeature weight.
            mus += paramsVector[crossmods]

        for varJ in np.arange(0, len(click_coefsv)):
            modulo = click_modulos[varJ]

            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[click_otherfeatureid[varJ]] * click_coefsv2[varJ]
            # all possible modality m
            mods = modalities * click_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)
            crossmods = (np.add.outer(modsJ, mods) % modulo) + click_offsets[varJ]
            lambdas += paramsVector[crossmods]

        mus = np.exp(mus)
        mus = mus / mus.sum(axis=1)[:, None]

        currentLambdas = lambdas[np.arange(len(lambdas)), x[varId]]
        lambdas -= currentLambdas[:, None]
        lambdas = np.exp(lambdas) * mus * py[:, None]

        for varJ in np.arange(0, len(disp_coefsv)):
            modulo = click_modulos[varJ]

            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
            # all possible modality m
            mods = modalities * disp_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)

            crossmods = (np.add.outer(modsJ, mods) % modulo) + disp_offsets[varJ]
            for i in range(0, nbsamples):
                results[crossmods[i]] += mus[i]
            # for i in range(0,nbsamples):
            #    xmods = mods  +  modsJ[i] + disp_offsets[varJ]
            #    results[xmods] += mus[i]

        for varJ in np.arange(0, len(click_coefsv)):
            modulo = click_modulos[varJ]
            modsJ = x[click_otherfeatureid[varJ]] * click_coefsv2[varJ]
            mods = modalities * click_coefsv[varJ]
            for i in range(0, nbsamples):
                xmods = (mods + modsJ[i]) % modulo + click_offsets[varJ]
                results[xmods] += lambdas[i]

        # lambdas = np.exp(lambdas)
        # mus = mus * (( lambdas.transpose() * y + 1-y )).transpose()
        # mus = mus * ( 1  +  lambdas) # buggycode
        # Sampling now modality of varId

        # varnew =  weightedSamples(mus)
        # updating the samples
        # x[varId] = varnew

    return results


def ComputeRWpred(self, samples=None, maxNbRows=1000, useNumba=True):
    (
        exportedDisplayWeights,
        exportedClickWeights,
        modalitiesByVarId,
        parameters,
    ) = self.exportWeightsAll()
    start = 0
    if samples is None:
        samples = self.samples
    py = samples.explambda / samples.expmu
    # py  = py /(1+py)

    rows = samples.data.transpose()
    starts = np.arange(0, len(rows), maxNbRows)
    slices = [(rows[start : start + maxNbRows], py[start : start + maxNbRows]) for start in starts]

    if useNumba:

        def myfun(s):
            return computeRaoBlackwellisedExpectations_numba(
                exportedDisplayWeights,
                exportedClickWeights,
                modalitiesByVarId,
                parameters,
                s[0],
                s[1],
            )

    else:

        def myfun(s):
            return computeRaoBlackwellisedExpectations(
                exportedDisplayWeights,
                exportedClickWeights,
                modalitiesByVarId,
                parameters,
                s[0],
                s[1],
            )

    runner = Parallel(n_jobs=14)
    jobs = [delayed(myfun)(myslice) for myslice in slices]
    predsSlices = runner(jobs)

    projection = np.array(predsSlices).sum(axis=0)

    projection = projection / samples.Size * np.exp(self.muIntercept)
    z0_on_z = 1 / np.mean((1 + samples.explambda / samples.expmu))  # = P(Y)
    projection *= z0_on_z * (1 + np.exp(self.lambdaIntercept))

    for w in self.displayWeights.values():
        if type(w.feature) is featuremappings.CrossFeaturesMapping:
            projection[w.indices] /= 2
    for var in self.clickWeights:
        w = self.clickWeights[var]
        if type(w.feature) is featuremappings.CrossFeaturesMapping:
            projection[w.indices] /= 2
        wd = self.displayWeights[var]
        projection[wd.indices] += projection[w.indices]

    return projection
    # samplesSlices = [ myfun(myslice) for myslice in  slices ]


@jit(nopython=True)
def computeRaoBlackwellisedExpectations_numba(
    exportedDisplayWeights, exportedClickWeights, modalitiesByVarId, paramsVector, x, py
):
    (
        allcoefsv,
        allcoefsv2,
        alloffsets,
        allotherfeatureid,
        allmodulos,
    ) = exportedDisplayWeights
    (
        click_allcoefsv,
        click_allcoefsv2,
        click_alloffsets,
        click_allotherfeatureid,
        click_allmodulos,
    ) = exportedClickWeights

    x = x.transpose()
    nbsamples = x.shape[1]

    results = np.zeros(len(paramsVector))

    # List of features ( one feature <=> its index in the arrays )
    f = np.arange(0, x.shape[0])
    #  Gibbs sampling may converge faster if the order in which we sample features is randomized.

    # For each feature, ressample this feature conditionally to the other
    for varId in f:

        # data describing the different crossfeatures involving varId  in " K(x).mu"
        # Those things are arrays, of len the number of crossfeatures involving varId.
        disp_coefsv = allcoefsv[varId]
        disp_coefsv2 = allcoefsv2[varId]
        disp_offsets = alloffsets[varId]
        disp_otherfeatureid = allotherfeatureid[varId]
        disp_modulos = allmodulos[varId]

        # idem, but crossfeatures in " K(x).theta" part of the model
        click_coefsv = click_allcoefsv[varId]
        click_coefsv2 = click_allcoefsv2[varId]
        click_offsets = click_alloffsets[varId]
        click_otherfeatureid = click_allotherfeatureid[varId]
        click_modulos = allmodulos[varId]

        # array of possible modalities of varId
        modalities = modalitiesByVarId[varId]  # Should be 0,1,2 ... NbModalities
        # for each modality, compute P( modality | other features ) as exp( dotproduct)
        # initializing dotproduct
        mus = np.zeros((nbsamples, len(modalities)))
        lambdas = np.zeros((nbsamples, len(modalities)))

        # Computing the dotproducts
        #  For each crossfeature containing varId
        for varJ in np.arange(0, len(disp_coefsv)):
            modulo = disp_modulos[varJ]
            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
            # all possible modality m
            mods = modalities * disp_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)
            crossmods = (AddOuter(modsJ, mods) % modulo) + disp_offsets[varJ]
            # Adding crossfeature weight.
            mus += getVectorValuesAtIndex(paramsVector, crossmods)
            # mus += paramsVector[crossmods]

        for varJ in np.arange(0, len(click_coefsv)):
            modulo = click_modulos[varJ]
            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[click_otherfeatureid[varJ]] * click_coefsv2[varJ]
            # all possible modality m
            mods = modalities * click_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)
            crossmods = (AddOuter(modsJ, mods) % modulo) + click_offsets[varJ]
            lambdas += getVectorValuesAtIndex(paramsVector, crossmods)

        mus = np.exp(mus)

        for i in range(0, mus.shape[0]):
            s = mus[i, :].sum()
            mus[i, :] /= s
        # mus = (mus / mus.sum(axis=1)[:,None]  )

        for i in range(0, mus.shape[0]):
            currentModality = x[varId, i]
            currentLambda = lambdas[i, currentModality]
            lambdas[i, :] -= currentLambda

        # currentLambdas = lambdas[np.arange(len(lambdas)), x[varId]]
        # lambdas -= currentLambdas[:,None]
        # lambdas = np.exp( lambdas) * mus  * py[:,None]
        lambdas = np.exp(lambdas) * mus
        for i in range(0, mus.shape[0]):
            lambdas[i, :] *= py[i]
        # lambdas *=  py[:,None]

        for varJ in np.arange(0, len(disp_coefsv)):
            modulo = disp_modulos[varJ]

            # let m a modality of feature varId, and m' a modality of the other feature
            #  Value of the crossfeature is " m *  disp_coefsv[varJ] + m' * disp_coefsv2[varJ]  "
            # values of m' in the data
            modsJ = x[disp_otherfeatureid[varJ]] * disp_coefsv2[varJ]
            # all possible modality m
            mods = modalities * disp_coefsv[varJ]
            # Computing crossfeatures
            # this is a matrix of shape (nbSamples, nbModalities of varId)

            crossmods = (AddOuter(modsJ, mods) % modulo) + disp_offsets[varJ]
            for i in range(0, nbsamples):
                results[crossmods[i]] += mus[i]
            # for i in range(0,nbsamples):
            #    xmods = mods  +  modsJ[i] + disp_offsets[varJ]
            #    results[xmods] += mus[i]

        for varJ in np.arange(0, len(click_coefsv)):
            modulo = click_modulos[varJ]
            modsJ = x[click_otherfeatureid[varJ]] * click_coefsv2[varJ]
            mods = modalities * click_coefsv[varJ]
            for i in range(0, nbsamples):
                xmods = (mods + modsJ[i]) % modulo + click_offsets[varJ]
                results[xmods] += lambdas[i]

        # lambdas = np.exp(lambdas)
        # mus = mus * (( lambdas.transpose() * y + 1-y )).transpose()
        # mus = mus * ( 1  +  lambdas) # buggycode
        # Sampling now modality of varId

        # varnew =  weightedSamples(mus)
        # updating the samples
        # x[varId] = varnew

    return results


@jit(nopython=True)
def AddOuter(x, y):
    r = np.zeros((len(y), len(x)), dtype="int32")
    r += x
    r = r.transpose()
    r += y
    return r


@jit(nopython=True)
def getVectorValuesAtIndex(x, indexingMatrix):
    r = np.zeros(indexingMatrix.shape, dtype="float64")
    for i in range(0, indexingMatrix.shape[0]):
        for j in range(0, indexingMatrix.shape[1]):
            r[i, j] = x[indexingMatrix[i, j]]
    return r
