import pandas as pd
import numpy as np
import scipy
import random
from collections import Counter
from joblib import Parallel, delayed
from typing import Dict
from aggregated_models.featuremappings import SingleFeatureMapping, CrossFeaturesMapping
from aggregated_models.SampleSet import SampleSet
from aggregated_models.SampleRdd import SampleRdd, VariableMRFParameters
from aggregated_models import featureprojections
from aggregated_models.baseaggmodel import BaseAggModel, WeightsSet
from aggregated_models import Optimizers
from aggregated_models.mrf_helpers import ComputeRWpred, fastGibbsSample, fastGibbsSampleFromPY0
import pyspark.sql.functions as F
import pyspark.sql as ps
import logging


_log = logging.getLogger(__name__)


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
        maxNbRowsPerSlice=50,
        sparkSession=None,
    ):
        super().__init__(aggdata, features)
        self.priorDisplays = priorDisplays
        self.exactComputation = exactComputation
        self.nbSamples = int(nbSamples) if not exactComputation else None

        self.regulL2 = regulL2

        self.displaysCfs = featureprojections.parseCFNames(self.features, displaysCfs)
        self.clicksCfs = featureprojections.parseCFNames(self.features, clicksCfs)

        self.allFeatures = self.features
        self.activeFeatures = activeFeatures if activeFeatures is not None else features
        # batch Size for the Gibbs sampler. (too high => memory issues on large models)
        self.maxNbRowsPerSlice = maxNbRowsPerSlice
        self.noiseDistribution = noiseDistribution
        self.sampleFromPY0 = sampleFromPY0
        # Compute Monte Carlo by sampling Y  (no good reason to do that ? )
        self.decollapseGibbs = False
        self.nbGibbsSteps = 1
        self.RaoBlackwellization = False
        self.regulL2Click = regulL2Click
        if regulL2Click is None:
            self.regulL2Click = regulL2
        self.lastPredict = None
        self.sparkSession = sparkSession
        # Preparing weights, parameters, samples ...

        self.modifiedGradient = False

        self.prepareFit()

    def setProjections(self):
        clickFeatures = self.features + self.clicksCfs
        self.clickProjections = {feature: self.aggdata.aggClicks[feature] for feature in clickFeatures}
        displayFeatures = self.features + self.displaysCfs
        self.displayProjections = {feature: self.aggdata.aggDisplays[feature] for feature in displayFeatures}
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
            [self.displayProjections[feature] for feature in self.features],
            self.nbSamples,
            self.decollapseGibbs,
            self.sampleFromPY0,
            self.maxNbRowsPerSlice,
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
        for feature in self.activeFeatures:
            weights = self.displayWeights[feature]
            proj = self.displayProjections[feature]
            self.parameters[weights.indices] = np.log(np.maximum(proj.Data, self.priorDisplays)) - logNbDisplay
            # init to log( P(v | C=0 ) instead ???

    def prepareFit(self):
        self.setProjections()  # building all weights and projections now
        self.setWeights()
        self.setActiveFeatures(self.activeFeatures)  # keeping only those active at the beginning
        self.initParameters()
        if self.sparkSession is not None:
            self.samples = self.buildSamplesRddFromSampleSet(self.samples)
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
        self.update()

    def predict_pandas(self, df, pred_col_name):
        _log.debug("Predicting on pandas Dataframe")
        # compute dot product on each line
        df["lambda"] = self.dotproductsOnDF(self.clickWeights, df) + self.lambdaIntercept
        df["mu"] = self.dotproductsOnDF(self.displayWeights, df) + self.muIntercept
        df["expmu"] = np.exp(df["mu"])
        df["explambda"] = np.exp(df["lambda"]) * df["expmu"]
        if "weight" in df.columns:
            df["E(NbNoClick)"] = df["expmu"] * df["weight"]
            df["E(NbClick)"] = df["explambda"] * df["weight"]
            df["E(NbDisplays)"] = df["E(NbClick)"] + df["E(NbNoClick)"]
        df[pred_col_name] = 1.0 / (1.0 + np.exp(-df["lambda"]))
        return df

    def _spark_dotproduct(self, df, weightsSet: Dict[str, WeightsSet]):
        single_features = [
            (w.offset, w.feature._fid) for w in weightsSet.values() if isinstance(w.feature, SingleFeatureMapping)
        ]
        cross_features = [
            (w.offset, w.feature._fid1, w.feature._fid2, w.feature.coefV2, w.feature.Modulo)
            for w in weightsSet.values()
            if isinstance(w.feature, CrossFeaturesMapping)
        ]

        variableMRFParameters = df.sql_ctx.sparkSession.sparkContext.broadcast(VariableMRFParameters(self.parameters))

        def compute_dot_prod(mods):
            result = 0
            for (o, fid) in single_features:
                result += variableMRFParameters.value.parameters[o + mods[fid]]
            for (o, fid1, fid2, coef, mod) in cross_features:
                result += variableMRFParameters.value.parameters[o + ((mods[fid1] + coef * mods[fid2]) % mod)]
            return result.item()

        return F.udf(compute_dot_prod)

    def predict_spark(self, df, prediction):
        _log.debug("Predicting on spark Dataframe")
        df = df.withColumn("mods", F.array(*[F.col(f) for f in self.features]))
        lamb = (self._spark_dotproduct(df, self.clickWeights)(F.col("mods")) + F.lit(self.lambdaIntercept)).alias(
            "lambda"
        )
        mu = (self._spark_dotproduct(df, self.displayWeights)(F.col("mods")) + F.lit(self.muIntercept)).alias("mu")
        expmu = F.exp(mu).alias("expmu")
        explambda = F.exp(lamb * expmu).alias("explambda")
        if "weight" in df.columns:
            weight = F.col("weight")
            e_noclick = (expmu * weight).alias("e_noclick")
            e_click = (explambda * weight).alias("e_click")
            e_nbdisplays = (e_noclick + e_click).alias("e_nbdisplays")
            df = df.withColumn("e_nbdisplays", e_nbdisplays)
        return df.withColumn(prediction, F.lit(1.0) / (F.lit(1.0) + F.exp(-lamb)))

    def predictDFinternal(self, df, prediction: str):
        if isinstance(df, pd.DataFrame):
            return self.predict_pandas(df, prediction)
        elif isinstance(df, ps.DataFrame):
            return self.predict_spark(df, prediction)
        else:
            raise NotImplementedError(f"Prediction not implemented for {type(df)}")

    def predictinternal(self, samples):
        samples.PredictInternal(self)

    def update(self):
        self.predictinternal(self.samples)

    def updateSamplesWithGibbs(self, samples):
        if not samples.allcrossmods:
            # Not applying Gibbs if full samples was generated
            samples.UpdateSampleWithGibbs(self)
        samples.UpdateSampleWeights(self)

    def getPredictionsVector(self, samples):
        if self.RaoBlackwellization:
            return ComputeRWpred(self, samples, self.maxNbRowsPerSlice)

        return samples.GetPrediction(self)

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
    def computeLoss_(self, samples=None, epsilon=1e-12):
        llh = self.computeLossNoRegul(samples, epsilon)
        regul = (self.parameters * self.parameters * self.regulVector).sum()
        return llh + regul / self.nbCoefs

    #  "approx" loss.
    def computeLossNoRegul_(self, samples=None, epsilon=1e-12):
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

        if self.modifiedGradient:
            displays = self.Data[: self.offsetClicks]
            clicks = self.Data[self.offsetClicks :]
            pdisplays = predictions[: self.offsetClicks]
            pclicks = predictions[self.offsetClicks :]
            ratio = (displays + 1) / (pdisplays + 1)
            smoothedClickGrad = -clicks * np.minimum(1 / ratio, 1.0) + pclicks * np.minimum(ratio, 1.0)
            gradient[self.offsetClicks :] = smoothedClickGrad

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
        df = self.predictDF(df, "prediction")
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

    def buildSamplesRddFromSampleSet(self, samples):
        return SampleRdd(
            samples.projections,
            self,
            self.sparkSession,
            samples.Size,
            samples.decollapseGibbs,
            samples.sampleFromPY0,
            samples.get_rows(),
            self.maxNbRowsPerSlice,
        )

    def buildSamplesSetFromSampleRdd(self, samples):
        return SampleSet(
            samples.projections,
            samples.Size,
            samples.decollapseGibbs,
            samples.sampleFromPY0,
            self.maxNbRowsPerSlice,
            samples.get_rows(),
        )

    def maxprobaratio(self, samples):
        probaSamples = samples.probaSamples
        pmodel = samples.expmu / np.exp(self.muIntercept)
        return max(pmodel / probaSamples)

    def fit(self, nbIter=100, alpha=0.01):
        Optimizers.simpleGradientStep(
            self,
            nbiter=nbIter,
            alpha=alpha,
            endIterCallback=lambda: self.updateAllSamplesWithGibbs(),
        )

    # export data useful to compute dotproduct
    def exportWeights(self, weights):
        weightsByVar = {}
        for feature in self.features:
            weightsByVar[feature] = [x for x in weights.values() if feature in x.feature.Name]
        allcoefsv = []
        allcoefsv2 = []
        alloffsets = []
        allotherfeatureid = []
        allmodulos = []

        for feature in self.features:
            coefsv = []
            coefsv2 = []
            offsets = []
            otherfeatureid = []
            modulos = []
            for w in weightsByVar[feature]:
                offsets += [w.offset]
                if type(w.feature) is CrossFeaturesMapping:
                    if feature == w.feature._v1:
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
            feature = self.features[i]
            modalitiesByVarId.append(np.arange(0, self.displayWeights[feature].feature.Size - 1))
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

        rows = samples.get_rows()

        starts = np.arange(0, len(rows), maxNbRows)
        slices = [(rows[start : start + maxNbRows], samples.y[start : start + maxNbRows]) for start in starts]
        nbGibbsSteps = self.nbGibbsSteps

        def myfun(s):
            s = fastGibbsSample(
                exportedDisplayWeights,
                exportedClickWeights,
                modalitiesByVarId,
                parameters,
                s[0],
                nbGibbsSteps,
                s[1],
            )
            return s

        def myfun_sampling_from_p_y0(s):
            s = fastGibbsSampleFromPY0(exportedDisplayWeights, modalitiesByVarId, parameters, s[0], nbGibbsSteps)
            return s

        if samples.sampleFromPY0:
            myfun = myfun_sampling_from_p_y0

        runner = Parallel(n_jobs=14)
        jobs = [delayed(myfun)(myslice) for myslice in slices]
        samplesSlices = runner(jobs)
        # samplesSlices = [ myfun(myslice) for myslice in  slices ]
        return np.vstack(samplesSlices).transpose()

    # Saving parameters and samples.
    # Warning: Not saving featuremappings,
    # would not work if instanciated from a different sample of the same dataset.
    def save(self, name):
        np.save(name + "_params", self.parameters)
        if type(self.samples) is SampleRdd:
            data = np.array(self.samples.rddSamples.collect()).transpose()
        else:
            data = self.samples.data
        np.save(name + "_samples", data)

    def load(self, name):
        extension = ".npy"
        data = np.load(name + "_samples" + extension)

        if type(self.samples) is SampleRdd:
            self.samples.rddSamples.unpersist()
            self.samples.rddSamples = self.sparkSession.sparkContext.parallelize(data.transpose())
        else:
            self.samples.data = data

        params = np.load(name + "_params" + extension)
        self.setparameters(params)
