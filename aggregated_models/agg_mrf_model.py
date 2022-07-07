import numba
import getpass
import json
import operator
import itertools
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pyspark.sql as ps
import pyspark.sql.functions as F
from dataclasses import dataclass, asdict
from joblib import Parallel, delayed
from typing import Dict

from aggregated_models.SampleSet import SampleSet, FullSampleSet
from aggregated_models.SampleRdd import SampleRdd
from aggregated_models.aggdataset import AggDataset
from aggregated_models.noiseDistributions import *
from aggregated_models import CrossFeaturesSet
from aggregated_models.baseaggmodel import BaseAggModel, WeightsSet
from aggregated_models import Optimizers
from aggregated_models.mrf_helpers import *
from aggregated_models.FeatureEncodings import *


import pyspark.sql.functions as F
import pyspark.sql as ps
import logging
# from thx.hadoop.spark_config_builder import SparkSession


_log = logging.getLogger(__name__)


@dataclass
class AggMRFModelParams:
    features: List[str]
    priorDisplays: float = 0.5  # used for initialization of weights associated to  "unknown modalities"
    exactComputation: bool = False  #
    nbSamples: float = 1e5  # Nb internal Gibbs samples
    regulL2: float = 1.0  # regularization parameter on 'mu'
    regulL2Click: Optional[float] = None  # regularization parameter on the parameters of P(Y|X). by default,
    # same value as regulL2
    displaysCfs: str = "*&*"
    clicksCfs: str = "*&*"
    sampleFromPY0: bool = False
    maxNbRowsPerSlice: int = 50
    nbGibbsIter: int = 1
    modifiedGradient: bool = True
    separateSamplesForNoise: bool = False
    useBetaPrior: bool = False
    projectNoiseGradient: bool = False
    modifiedGradientForNoise: bool = True
    gaussiansigma: float = 0
    gibbsMaxNbModalities: int = 1
    sampleMissingModalityInLearning: bool = True

    muStepSizeMultiplier: float = None
    maxNbIters: int = 1000

    # current nb iterations of the model. Updated during training.
    nbIters: int = 0

    multinomialRescaling: bool = False


class AggMRFModel(BaseAggModel):
    def __init__(
        self,
        aggdata: AggDataset,
        config_params: AggMRFModelParams,
        sparkSession = None, # Optional[SparkSession]
    ):
        super().__init__(aggdata, config_params.features)
        self.config_params = config_params
        self.priorDisplays = config_params.priorDisplays
        self.exactComputation = config_params.exactComputation
        self.nbSamples = int(config_params.nbSamples) if not config_params.exactComputation else None

        self.regulL2 = config_params.regulL2
        self.gibbsMaxNbModalities = config_params.gibbsMaxNbModalities

        self.displaysCfs = CrossFeaturesSet.parseCFNames(self.features, config_params.displaysCfs)
        self.clicksCfs = CrossFeaturesSet.parseCFNames(self.features, config_params.clicksCfs)

        # batch Size for the Gibbs sampler. (too high => memory issues on large models)
        self.maxNbRowsPerSlice = config_params.maxNbRowsPerSlice
        # self.noiseDistribution = config_params.noiseDistribution
        self.sampleFromPY0 = config_params.sampleFromPY0
        # Compute Monte Carlo by sampling Y  (no good reason to do that ? )

        self.regulL2Click = config_params.regulL2Click
        if config_params.regulL2Click is None:
            self.regulL2Click = config_params.regulL2

        self.sparkSession = sparkSession
        # Preparing weights, parameters, samples ...

        self.modifiedGradient = config_params.modifiedGradient
        self.modifiedGradientForNoise = config_params.modifiedGradientForNoise

        self.nbGibbsIter = config_params.nbGibbsIter
        self.gaussiansigma = config_params.gaussiansigma
        self.noiseDistribution = None
        if self.gaussiansigma != 0:
            self.noiseDistribution = GaussianNoise(self.gaussiansigma)

        self.prepareFit()
        self.Data = self.DataRemoveNegatives

        self.fitOnlyMu = False

    def setWeights(self):
        self.displayWeights, self.offsetClicks = self.prepareWeights(self.features + self.displaysCfs)
        self.clickWeights, offset = self.prepareWeights(self.features + self.clicksCfs, self.offsetClicks)
        self.offset = offset
        self.parameters = np.zeros(offset)
        self.setRegul(self.regulL2Click, self.regulL2)

    def setRegul(self, regulL2Click, regulL2):
        self.regulL2Click = regulL2Click
        self.regulL2 = regulL2

    @property
    def nbIters(self):
        return self.config_params.nbIters

    @property
    def regulVector(self):
        regulVector = np.zeros(len(self.parameters)) + self.regulL2
        regulVector[self.offsetClicks :] = self.regulL2Click
        return regulVector

    def buildSamples(self):
        if self.sparkSession is None:
            if not self.exactComputation:
                samples = SampleSet(
                    self.displaySimpleProjections,
                    self.nbSamples,
                    self,
                    self.sampleFromPY0,
                    self.maxNbRowsPerSlice,
                )
            else:
                samples = FullSampleSet(self.displaySimpleProjections, self, self.maxNbRowsPerSlice)
        else:
            variableMRFParameters, constantMRFParameters = self._get_mrf_parameters()
            samples = SampleRdd(
                [self.displayProjections[feature] for feature in self.features],
                self.sparkSession,
                constantMRFParameters,
                variableMRFParameters,
                self.nbSamples,
                False,
                self.sampleFromPY0,
                self.maxNbRowsPerSlice,
            )
        return samples

    def _get_mrf_parameters(self):
        variableMRFParameters = VariableMRFParameters(self)
        (exportedDisplayWeights, exportedClickWeights, modalitiesByVarId, _) = self.exportWeightsAll()
        constantMRFParameters = ConstantMRFParameters(
            self.nbSamples,
            self.parameters.size,
            self.sampleFromPY0,
            exportedDisplayWeights,
            exportedClickWeights,
            self.displayWeights,
            self.clickWeights,
            modalitiesByVarId,
            self.muIntercept,
            self.lambdaIntercept,
        )
        return variableMRFParameters, constantMRFParameters

    def setSamples(self):
        self.samples = self.buildSamples()
        if self.config_params.separateSamplesForNoise:
            self.samplesForNoise = self.buildSamples()

    @property
    def displayProjections(self):
        return self.aggdata.aggDisplays

    @property
    def clickProjections(self):
        return self.aggdata.aggDisplays

    @property
    def displaySimpleProjections(self):
        return [self.displayProjections[feature] for feature in self.features]

    @property
    def nbCoefs(self):
        return len(self.Data)

    @property
    def mu(self):
        return self.parameters[: self.offsetClicks]

    @property
    def theta(self):
        return self.parameters[self.offsetClicks : self.offset]

    @property
    def aggdisplaysvector(self):
        return self.Data[: self.offsetClicks]

    @property
    def aggclicksvector(self):
        return self.Data[self.offsetClicks : self.offset]

    @property
    def pData(self):
        return self.getPredictionsVector(self.samples)

    @property
    def paggdisplaysvector(self):
        return self.pData[: self.offsetClicks]

    @property
    def paggclicksvector(self):
        return self.pData[self.offsetClicks : self.offset]

    @property
    def currentParams(self):
        return self.parameters

    def initParameters(self):
        v0 = self.features[0]
        self.normgrad = 1.0
        nbclicks = self.aggdata.Nbclicks
        nbdisplays = self.aggdata.Nbdisplays
        self.muIntercept = np.log(nbdisplays - nbclicks)
        self.lambdaIntercept = np.log(nbclicks) - self.muIntercept
        logNbDisplay = np.log(nbdisplays)
        for feature in self.features:
            weights = self.displayWeights[feature]
            proj = self.displayProjections[feature]
            self.parameters[weights.indices] = np.log(np.maximum(proj.Data, self.priorDisplays))
            self.parameters[weights.indices] -= self.parameters[weights.indices].mean()

    def prepareFit(self):
        self.setWeights()
        self.setAggDataVector()
        self.initParameters()
        self.setSamples()  # reseting data
        self.update()
        return

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
            (w.offset, w.feature._fid1, w.feature._fid2, w.feature.coefV2, w.feature.Size)
            for w in weightsSet.values()
            if isinstance(w.feature, CrossFeatureEncoding)
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
        if self.config_params.separateSamplesForNoise:
            self.predictinternal(self.samplesForNoise)

    def updateSamplesWithGibbs(self, samples):
        samples.UpdateSampleWithGibbs(self, self.nbGibbsIter)

    def getPredictionsVector(self, samples):
        return samples.GetPrediction(self) * self.aggdata.Nbdisplays

    def getPredictionsVector_(self, samples, index):
        x = self.parameters * 0
        for w in self.displayWeights.values():
            x[w.indices] = w.feature.Project_(samples.data[:, index], samples.pdisplays[index])  # Correct for grads
        for w in self.clickWeights.values():
            x[w.indices] = w.feature.Project_(samples.data[:, index], samples.Eclick[index])
        return x

    def setAggDataVector(self):
        self.Data = self.getAggDataVector(self.clickWeights, self.aggdata.aggClicks)
        self.Data += self.getAggDataVector(self.displayWeights, self.aggdata.aggDisplays)

    @property
    def DataRemoveNegatives(self):
        return self.Data * (self.Data > 0)

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

        try:
            predictions = self.getPredictionsVector(samples)
        except:
            print("exception while computing predictions. Retrying:")
            predictions = self.getPredictionsVector(samples)

        noise = self.Data * 0
        if self.noiseDistribution is not None:
            if self.config_params.separateSamplesForNoise:
                noise = self.expectedNoise(None, self.samplesForNoise)
            else:
                noise = self.expectedNoise(predictions, samples)

            if self.config_params.projectNoiseGradient:
                proj = self.project(self.Data - noise)
                noise = self.Data - proj

        gradient = -self.Data + predictions + noise  # - (data-noise - preds)

        if self.modifiedGradient:
            for f in self.clickWeights:
                wc = self.clickWeights[f]
                wd = self.displayWeights[f]

                displays = self.Data[wd.indices]
                clicks = self.Data[wc.indices]
                if self.modifiedGradientForNoise:
                    displays -= noise[wd.indices]
                    clicks -= noise[wc.indices]
                    displays[displays < 0] = 0  # Should not happen ?
                    clicks[clicks < 0] = 0

                pdisplays = predictions[wd.indices]
                pclicks = predictions[wc.indices]

                ratio = (displays + 1) / (pdisplays + 1)
                smoothedClickGrad = -clicks * np.minimum(1 / ratio, 1.0) + pclicks * np.minimum(ratio, 1.0)
                gradient[wc.indices] = smoothedClickGrad

        gradient += 2 * self.parameters * self.regulVector

        if self.fitOnlyMu:
            for f in self.clickWeights:
                wc = self.clickWeights[f]
                gradient[wc.indices] = 0

        self.normgrad = sum(gradient * gradient)
        return gradient

    # Estimating E( L | D = data ,parmeters )  where L ~ Laplace and D = L + AggregatedCounts
    def expectedNoise(self, predictions, samples):
        # approximation: assuming that (Li) are independent knowing D.
        return self.expectedNoiseIndepApprox(predictions, samples)

    def expectedNoiseIndepApprox(self, predictions, samples=None):
        if predictions is None:
            predictions = self.getPredictionsVector(samples)
        if self.config_params.useBetaPrior:
            return expectedGaussianKnowingDataPlusNoiseAndSampledDataExpect(
                self.Data, predictions, self.gaussiansigma, self.aggdata.Nbdisplays, self.config_params.nbSamples
            )
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
        if self.config_params.separateSamplesForNoise:
            self.updateSamplesWithGibbs(self.samplesForNoise)

    def buildSamplesRddFromSampleSet(self, samples):
        return self.buildSamplesRddFromData(samples.get_rows())

    def buildSamplesRddFromData(self, data):
        variableMRFParameters, constantMRFParameters = self._get_mrf_parameters()
        return SampleRdd(
            self.displaySimpleProjections,
            self.sparkSession,
            constantMRFParameters,
            variableMRFParameters,
            data.shape[0],
            False,
            self.sampleFromPY0,
            self.maxNbRowsPerSlice,
            data,
        )

    def buildSamplesSetFromSampleRdd(self, samples):
        return SampleSet(
            self.displaySimpleProjections,
            samples.Size,
            self,
            samples.sampleFromPY0,
            self.maxNbRowsPerSlice,
            samples.get_rows(),
        )

    def fit(self, nbIter=100, alpha=None):
        if alpha is None:
            alpha = 0.5 / len(self.clickWeights)
        if self.config_params.muStepSizeMultiplier is not None:
            alpha = np.ones(len(self.parameters)) * alpha
            alpha[: self.offsetClicks] *= self.config_params.muStepSizeMultiplier

        def endIterCallback():
            self.config_params.nbIters += 1
            self.updateAllSamplesWithGibbs()

        Optimizers.simpleGradientStep(
            self,
            nbiter=nbIter,
            alpha=alpha,
            endIterCallback=endIterCallback,
        )

    # export data useful to compute dotproduct
    def exportWeights(self, weights):
        weightsByVar = {}
        for feature in self.features:
            weightsByVar[feature] = [x for x in weights.values() if feature in x.feature.Name.split("&")]
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
                if type(w.feature) is CrossFeatureEncoding:
                    if feature == w.feature._v1:
                        coefsv += [1]
                        coefsv2 += [w.feature.coefV2]
                        otherfeatureid += [self.features.index(w.feature._v2)]
                        modulos += [w.feature.Size]
                        # formula for indice of the crossmodality (V1=v1,V2=v2) in the parameter vector is :
                        #  w.offset + v1 * coefsv + v2 * coefsv2
                    else:
                        coefsv += [w.feature.coefV2]
                        coefsv2 += [1]
                        otherfeatureid += [self.features.index(w.feature._v1)]
                        modulos += [w.feature.Size]

                else:
                    coefsv += [1]
                    coefsv2 += [0]
                    otherfeatureid += [0]
                    modulos += [w.feature.Size]

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
            if self.config_params.sampleMissingModalityInLearning:
                modalitiesByVarId.append(np.arange(0, self.displayWeights[feature].feature.Size))
            else:
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
        nbGibbsSteps = self.nbGibbsIter
        if nbGibbsSteps == 0:
            return samples.columns
        samples.sampleY()
        (
            exportedDisplayWeights,
            exportedClickWeights,
            modalitiesByVarId,
            parameters,
        ) = self.exportWeightsAll()

        parameters = self.currentParams

        start = 0
        rows = samples.get_rows()
        starts = np.arange(0, len(rows), maxNbRows)
        slices = [(rows[start : start + maxNbRows], samples.y[start : start + maxNbRows]) for start in starts]

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

    def pysparkGibbsSampler(self, samples, nbGibbsIter=1):
        if self.sampleFromPY0:
            return self._runGibbsSampling_fromPY0(samples, nbGibbsIter)
        else:
            return self.pysparkGibbsSampler_samplingY(samples, nbGibbsIter)

    def pysparkGibbsSampler_samplingY(model, self, nbGibbsIter=1):
        constantMRFParameters = self.constantMRFParameters
        variableMRFParameters = self.variableMRFParameters

        def expdotproducts(x):
            lambdas = 0
            for w in constantMRFParameters.value.clickWeights.values():
                lambdas += variableMRFParameters.value.parameters[w.feature.Values_(x) + w.offset]
            explambda = np.exp(lambdas + constantMRFParameters.value.lambdaIntercept)
            return x, explambda

        def sampleY(x_expdotprod):
            x, expdotprod = x_expdotprod
            py = expdotprod / (1 + expdotprod)
            y = 1 if py < np.random.random() else 0
            return x, y

        rdd_x_expdotprod = self.rdd.map(expdotproducts)
        rdd_x_y = rdd_x_expdotprod.map(sampleY)
        gibbsMaxNbModalities = model.gibbsMaxNbModalities

        def myfun_sampling(sample_y):
            sample, y = sample_y
            if y == 0:
                parameters = variableMRFParameters.value.parameters
            else:
                parameters = variableMRFParameters.value.parametersForPY1
            return blockedGibbsSampler_PY0(
                constantMRFParameters.value.explosedDisplayWeights,
                constantMRFParameters.value.modalitiesByVarId,
                parameters,
                sample,
                nbGibbsIter,
                gibbsMaxNbModalities,
            )

        return rdd_x_y.map(myfun_sampling)

    def _runGibbsSampling_fromPY0(model, self, nbGibbsIter=1):
        constantMRFParameters = self.constantMRFParameters
        variableMRFParameters = self.variableMRFParameters
        gibbsMaxNbModalities = model.gibbsMaxNbModalities

        def myfun_sampling_from_p_y0(sample):
            return blockedGibbsSampler_PY0(
                constantMRFParameters.value.explosedDisplayWeights,
                constantMRFParameters.value.modalitiesByVarId,
                variableMRFParameters.value.parameters,
                sample,
                nbGibbsIter,
                gibbsMaxNbModalities,
            )

        return self.rddSamples.map(myfun_sampling_from_p_y0)

    # Saving parameters and samples.
    # Warning: Not saving featuremappings,
    # would not work if instanciated from a different sample of the same dataset.
    def save(self, base_local_dir: str, base_hdfs_dir: Optional[str] = None):

        if base_hdfs_dir is None:
            base_hdfs_dir = "/user/" + getpass.getuser() + "/" + base_local_dir
            # print(base_hdfs_dir)

        base_local_path = Path(base_local_dir)
        base_hdfs_path = Path(base_hdfs_dir)
        base_local_path.mkdir(parents=True, exist_ok=True)

        with open(base_local_path / "config_params.json", "w") as config_fp:
            json.dump(asdict(self.config_params), config_fp)

        with open(base_local_path / "aggdata", "wb") as aggdata_fp:
            self.aggdata.dump(aggdata_fp)

        np.save(str(base_local_path / "parameters"), self.parameters)
        try:
            sparkdf = self.samples.rddSamples.map(lambda x: [int(i) for i in x]).toDF()
            sparkdf.write.mode("overwrite").parquet(str(base_hdfs_path / "samples"))
        except:
            print("fail to save samples with spark. maybe running in memory?")
            try:
                np.save(str(base_local_path / "samples"), self.samples.columns)
            except:
                print("Failed to save samples")

    @staticmethod
    def load(
        base_local_dir: str,
        base_hdfs_dir: Optional[str] = None,
        spark_session = None, # Optional[SparkSession]
        loadSamples: bool = True,
    ):

        if base_hdfs_dir is None and loadSamples:
            base_hdfs_dir = "/user/" + getpass.getuser() + "/" + base_local_dir
            print(f"using base_hdfs_dir = {base_hdfs_dir} ")

        base_local_path = Path(base_local_dir)
        base_hdfs_path = Path(base_hdfs_dir)

        with open(base_local_path / "config_params.json", "r") as config_fp:
            config_params = AggMRFModelParams(**json.load(config_fp))

        with (base_local_path / "aggdata").open("rb") as aggdata_fp:
            agg_dataset = AggDataset.load(aggdata_fp)

        model = AggMRFModel(agg_dataset, config_params=config_params, sparkSession=spark_session)

        params = np.load(str(base_local_path / "parameters.npy"))
        model.setparameters(params)
        if loadSamples:
            if spark_session:
                print(f"loading samples from parquet at {str(base_hdfs_path / 'samples')}")
                rdd = spark_session.read.parquet(str(base_hdfs_path / "samples")).rdd
                model.samples.nbSamples = rdd.count()
                nbpartitions = int(model.samples.nbSamples / model.maxNbRowsPerSlice)
                print(f"nbsamples:{model.samples.nbSamples}   nbpartitions:{nbpartitions}")
                model.samples.rddSamples = rdd.repartition(nbpartitions).map(list)
            else:
                samplesfile = str(base_local_path / "samples.npy")
                try:
                    model.samples.columns = np.load(samplesfile)
                except:
                    print("WARNING: could not load samples from " + samplesfile)
        return model

    def pysparkPredict(self, samples):

        constantMRFParameters = samples.constantMRFParameters
        variableMRFParameters = samples.variableMRFParameters
        sampleFromPY0 = self.sampleFromPY0

        def compute_explambda(x):
            lambdas = 0
            for w in constantMRFParameters.value.clickWeights.values():
                lambdas += variableMRFParameters.value.parameters[w.feature.Values_(x) + w.offset]
            explambda = np.exp(lambdas + constantMRFParameters.value.lambdaIntercept)
            return explambda

        def prepareX(x):
            explambda = compute_explambda(x)

            py = explambda / (1 + explambda)
            w = (
                1 + explambda if sampleFromPY0 else 1
            )  # importance weight (Unormalized, we normalize later by the average of the weights)
            proj_display = oneHotEncode(x, constantMRFParameters.value.explosedDisplayWeights)
            proj_click = oneHotEncode(x, constantMRFParameters.value.explosedClickWeights)

            return (
                (proj_display, w),
                (proj_click, py * w),
                ([-1], w),  # Keeping index '-1' to get sums of edisplays for normalization
            )

        rdd_keys_value = samples.rdd.flatMap(prepareX)

        def explodeProjections(modalities_value):
            modalities, value = modalities_value
            return zip(modalities, itertools.repeat(value))

        rdd_key_value = rdd_keys_value.flatMap(explodeProjections)
        rdd_reduced = rdd_key_value.reduceByKey(operator.add)
        projections = rdd_reduced.collectAsMap()
        return samples._rebuid_predictions(self, projections)

    @staticmethod
    def gradconstraints_varI(x, exportedDisplayWeights, modalitiesByVarId, varI):
        (
            allcoefsv,
            allcoefsv2,
            alloffsets,
            allotherfeatureid,
            allmodulos,
        ) = exportedDisplayWeights
        grad = x * 0
        disp_coefsv = allcoefsv[varI]
        disp_coefsv2 = allcoefsv2[varI]
        disp_offsets = alloffsets[varI]
        disp_otherfeatureid = allotherfeatureid[varI]
        disp_modulos = allmodulos[varI]

        modalities = modalitiesByVarId[varI]
        indices_simple = modalities + alloffsets[varI][0]

        for j in np.arange(0, len(disp_coefsv)):
            varJ = disp_otherfeatureid[j]
            modulo = disp_modulos[j]
            modsJ = modalitiesByVarId[varJ] * disp_coefsv2[j]
            mods = modalitiesByVarId[varI] * disp_coefsv[j]
            crossmods = (np.add.outer(modsJ, mods) % modulo) + disp_offsets[j]

            error = x[crossmods].sum(axis=0) - x[indices_simple]
            error /= len(modsJ)
            grad[indices_simple] += error
            grad[crossmods] -= error[np.newaxis, :]
        return grad

    @staticmethod
    def gradconstraints(x, exportedDisplayWeights, modalitiesByVarId):
        grad = x * 0
        fids = np.arange(0, len(exportedDisplayWeights[0]))
        for i in fids:
            grad += AggMRFModel.gradconstraints_varI(x, exportedDisplayWeights, modalitiesByVarId, i)
        return grad

    def project(self, x, nbiters=10, alpha=0.9):
        (exportedDisplayWeights, exportedClickWeights, modalitiesByVarId, _) = self.exportWeightsAll()
        for i in range(0, nbiters):
            g = AggMRFModel.gradconstraints(x, exportedDisplayWeights, modalitiesByVarId)
            gc = AggMRFModel.gradconstraints(x, exportedClickWeights, modalitiesByVarId)

            g = g / (1 + np.linalg.norm(g))
            gc = gc / (1 + np.linalg.norm(gc))
            x = x - g * g.dot(x) * alpha
            x = x - gc * gc.dot(x) * alpha

            x[x < 0] = 0
        return x
