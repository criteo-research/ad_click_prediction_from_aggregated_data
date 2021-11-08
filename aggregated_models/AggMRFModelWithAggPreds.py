import numpy as np
import pandas as pd

from aggregated_models.agg_mrf_model import *
from aggregated_models.aggdataset import AggDataset, FeaturesSet

from thx.hadoop.spark_config_builder import SparkSession
from joblib import Parallel, delayed
from aggregated_models.mrf_helpers import (
    fastGibbsSampleFromPY0_withAggPredictions,
)

class AggMRFModelWithAggPreds(AggMRFModel):
    def __init__(
        self,
        aggdata: AggDataset,
        config_params: AggMRFModelParams,
        theta0,
        sparkSession: Optional[SparkSession] = None,
        linkFunctionId: int = 1,  #  0 -> Identity  , 1 -> sigmoid, 2 -> exp
    ):
        self.theta0 = theta0
        self.linkFunctionId = linkFunctionId
        super().__init__(aggdata, config_params, sparkSession)

 
    def setWeights(self):
        super().setWeights()
        self.offsetNu = len(self.parameters)
        self.parameters = np.hstack([self.parameters, np.zeros(len(self.mu))])

    def setAggDataVector(self):
        super().setAggDataVector()
        dataNu = self.getAggDataVector(self.displayWeights, self.aggdata.aggregations["p0"])[: len(self.theta0)]
        self.Data[self.offsetNu :] = dataNu

    @property
    def nu(self):
        return self.parameters[self.offsetNu :]

    @property
    def aggp0vector(self):
        return self.Data[self.offsetNu :]

    @property
    def paggp0vector(self):
        return self.getPredictionsVector(self.samples)[self.offsetNu :]

    @property
    def regulVector(self):
        regulVector = np.zeros(len(self.parameters)) + self.regulL2
        regulVector[self.offsetClicks : self.offsetNu] = self.regulL2Click
        regulVector[self.offsetNu :] = self.regulL2Click
        return regulVector

    def pysparkPredict(self, samples):
        constantMRFParameters = samples.constantMRFParameters
        variableMRFParameters = samples.variableMRFParameters
        linkFunctionId = self.linkFunctionId

        def compute_explambda(x):
            lambdas = 0
            for w in constantMRFParameters.value.clickWeights.values():
                lambdas += variableMRFParameters.value.parameters[w.feature.Values_(x) + w.offset]
            explambda = np.exp(lambdas + constantMRFParameters.value.lambdaIntercept)
            return explambda

        def compute_baselineprediction(x):
            xDotTheta0 = 0
            for w in constantMRFParameters.value.displayWeights.values():
                xDotTheta0 += variableMRFParameters.value.theta0[w.feature.Values_(x) + w.offset]
            if linkFunctionId == 0:
                prediction = xDotTheta0
            elif linkFunctionId == 1:
                prediction = 1 / (1 + np.exp(-xDotTheta0))
            elif linkFunctionId == 2:
                prediction = np.exp(-xDotTheta0)
            return prediction

        def prepareX(x):
            explambda = compute_explambda(x)
            baselinepred = compute_baselineprediction(x)
            w = 1 + explambda
            proj_display = oneHotEncode(x, constantMRFParameters.value.explosedDisplayWeights)
            proj_click = oneHotEncode(x, constantMRFParameters.value.explosedClickWeights)
            proj_display_nu = proj_display + variableMRFParameters.value.offsetNu
            return (
                (proj_display, w),
                (proj_click, explambda),  # * 1 / (1 + explambda) * w  -> simlifies out
                (proj_display_nu, baselinepred * w),
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

    def pysparkGibbsSampler(self, samples, nbGibbsIter=1):
        constantMRFParameters = samples.constantMRFParameters
        variableMRFParameters = samples.variableMRFParameters

        def compute_kx_dot_theta0(x):
            params_theta0 = variableMRFParameters.value.theta0
            xDotTheta0 = 0
            for w in constantMRFParameters.value.displayWeights.values():
                xDotTheta0 += params_theta0[w.feature.Values_(x) + w.offset]
            return xDotTheta0

        def compute_kx_dot_nu(x):
            params_nu = variableMRFParameters.value.nu
            xDotNu = 0
            for w in constantMRFParameters.value.displayWeights.values():
                xDotNu += params_nu[w.feature.Values_(x) + w.offset]
            return xDotNu

        linkFunctionId = self.linkFunctionId

        def sampling(x):
            kx_dot_theta0 = compute_kx_dot_theta0(x)
            kx_dot_nu = compute_kx_dot_nu(x)
            return fastGibbsSampleFromPY0_withAggPredictions(
                constantMRFParameters.value.explosedDisplayWeights,
                constantMRFParameters.value.modalitiesByVarId,
                variableMRFParameters.value.parameters,
                variableMRFParameters.value.theta0,
                variableMRFParameters.value.nu,
                x,
                kx_dot_theta0,
                kx_dot_nu,
                nbGibbsIter,
                linkFunctionId,
            )

        return samples.rdd.map(sampling)


    def RunParallelGibbsSampler(self, samples, maxNbRows=1000):
        (
            exportedDisplayWeights,
            exportedClickWeights,
            modalitiesByVarId,
            parameters,
        ) = self.exportWeightsAll()

        params_theta0 = self.theta0.copy()
        params_vu = self.nu.copy()
        rows = samples.data.transpose()

        kx_dot_theta0 = self.dotproducts_(self.displayWeights, rows.transpose(), params_theta0)
        kx_dot_vu = self.dotproducts_(self.displayWeights, rows.transpose(), params_vu)

        start = 0
        starts = np.arange(0, len(rows), maxNbRows)
        i = np.arange(0, len(rows))
        slicesIndices = [i[start : start + maxNbRows] for start in starts]
        slices = [(rows[ind], kx_dot_theta0[ind], kx_dot_vu[ind]) for ind in slicesIndices]
        nbGibbsIter = self.nbGibbsIter

        linkFunctionId = self.linkFunctionId

        def myfun(s):
            x = s[0]
            kx_dot_theta0 = s[1]
            kx_dot_vu = s[2]
            n = len(kx_dot_vu)
            for i in np.arange(0, n):
                x[i] = fastGibbsSampleFromPY0_withAggPredictions(
                    exportedDisplayWeights,
                    modalitiesByVarId,
                    parameters,
                    params_theta0,
                    params_vu,
                    x[i],
                    kx_dot_theta0[i],
                    kx_dot_vu[i],
                    nbGibbsIter,
                    linkFunctionId,
                )
            return x

        if True:
            runner = Parallel(n_jobs=14)
            jobs = [delayed(myfun)(myslice) for myslice in slices]
            samplesSlices = runner(jobs)
        else:
            samplesSlices = [myfun(myslice) for myslice in slices]

        return np.vstack(samplesSlices).transpose()
