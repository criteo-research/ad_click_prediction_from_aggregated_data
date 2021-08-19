import itertools
import operator
from typing import List, Optional

import numpy as np
from pyspark import Broadcast
from pyspark.rdd import RDD
import bisect

from pyspark.sql.session import SparkSession
from aggregated_models.mrf_helpers import (
    ConstantMRFParameters,
    VariableMRFParameters,
    gibbsOneSampleFromPY0,
    oneHotEncode,
    blockedGibbsSampler_PY0,
)

MAXMODALITIES = 1e7


# set of samples of 'x' used internally by AggMRFModel
class SampleRdd:
    def __init__(
        self,
        projections,
        sparkSession,
        constantModelParameters: ConstantMRFParameters = None,
        variableModelParameters: VariableMRFParameters = None,
        nbSamples=100,
        decollapseGibbs=False,
        sampleFromPY0=False,
        maxNbRowsPerSlice=500,
        data=None,
    ):
        self.projections = projections
        self.decollapseGibbs = decollapseGibbs
        self.sampleFromPY0 = sampleFromPY0
        self.features = [p.feature for p in projections]
        self.featurenames = [f.Name for f in self.features]
        self.Size = nbSamples
        self.allcrossmods = False
        self.prediction: Optional[np.ndarray] = None
        self.sparkSession: SparkSession = sparkSession
        self.broadcast_history: List[Broadcast] = list()
        if data is not None:
            self.rddSamples: RDD = sparkSession.sparkContext.parallelize(data, numSlices=nbSamples / maxNbRowsPerSlice)
        else:
            self.rddSamples = self._buildRddSamples(maxNbRowsPerSlice)
        if constantModelParameters is not None and variableModelParameters is not None:
            self._broadcast_parameters(constantModelParameters, variableModelParameters)

    def _broadcast_parameters(self, constantModelParameters, variableModelParameters):
        self.variableMRFParameters = self.sparkSession.sparkContext.broadcast(variableModelParameters)
        self.constantMRFParameters = self.sparkSession.sparkContext.broadcast(constantModelParameters)

    def _buildRddSamples(self, maxNbRowsPerSlice: int) -> RDD:
        dummySampleRdd = self.sparkSession.sparkContext.parallelize(
            np.ones(self.Size), numSlices=self.Size / maxNbRowsPerSlice
        )

        vectorSize = len(self.projections)

        cum_probas_list = list()
        for projection in self.projections:
            counts = projection.Data
            probas = counts / sum(counts)
            cumprobas = np.cumsum(probas)
            cum_probas_list.append(cumprobas)

        cumprobas_broadcast = self.sparkSession.sparkContext.broadcast(cum_probas_list)
        self.broadcast_history.append(cumprobas_broadcast)

        def _buildIndependant(row):
            cumprobas = cumprobas_broadcast.value
            rand = np.random.random_sample(vectorSize)
            return np.array([bisect.bisect(cumprobas[i], r) for i, r in enumerate(rand)])

        initializedSampleRdd = dummySampleRdd.map(_buildIndependant)
        # caching before checkpoint, because checkpoint would trigger the computation twice.
        initializedSampleRdd.cache()
        initializedSampleRdd.checkpoint()
        return initializedSampleRdd

    def get_rows(self):
        data = self.rddSamples.collect()
        return np.array([d for d in data])

    def UpdateSampleWithGibbs(self, model, nbGibbsIter=1):

        if nbGibbsIter > 0:
            rddnew = self._runGibbsSampling(model, nbGibbsIter)
            rddnew.cache()  # caching before checkpoint, because checkpoint would triger the computation twice.
            rddnew.checkpoint()
            rddnew.count()  # force compute before uncache
            try:
                self.rddSamples.unpersist()  # clean previous rdd
            except:
                print("error in self.rddSamples.unpersist() ")  # usually "Futures timed out..."

            self.rddSamples = rddnew
            self.cleanBroadcasts()

    def cleanBroadcasts(self):
        for k in self.broadcast_history:
            # after the checkpoint, we can safely delete old broadcasted parameters
            try:
                k.destroy()  # sometimes crashing with a "timeout". seems to happen when one of the nodes of the sesssion is having issues (memory or other ?)
            except:
                print("Failure while trying to destroy a broadcast")
        self.broadcast_history = list()

    def _runGibbsSampling(self, model, nbGibbsIter=1):
        if self.sampleFromPY0:
            return self._runGibbsSampling_fromPY0(model, nbGibbsIter)
        else:
            return self._runGibbsSampling_samplingY(model, nbGibbsIter)

    def _runGibbsSampling_samplingY(self, model, nbGibbsIter=1):
        constantMRFParameters = self.constantMRFParameters
        variableMRFParameters = self.variableMRFParameters

        rdd_x_expdotprod = self._compute_rdd_expdotproducts(model)

        def sampleY(x_expdotprod):
            x, expdotprod = x_expdotprod
            py = expdotprod / (1 + expdotprod)
            y = 1 if py < np.random.random() else 0
            return x, y

        rdd_x_y = rdd_x_expdotprod.map(sampleY)

        gibbsMaxNbModalities = model.gibbsMaxNbModalities
        if gibbsMaxNbModalities > 1:

            def _sampling(sample, parameters):
                return blockedGibbsSampler_PY0(
                    constantMRFParameters.value.explosedDisplayWeights,
                    constantMRFParameters.value.modalitiesByVarId,
                    parameters,
                    sample,
                    nbGibbsIter,
                    gibbsMaxNbModalities,
                )
                return new_sample

        else:

            def _sampling(sample, parameters):
                return gibbsOneSampleFromPY0(
                    constantMRFParameters.value.explosedDisplayWeights,
                    constantMRFParameters.value.modalitiesByVarId,
                    parameters,
                    sample,
                    nbGibbsIter,
                )

        def myfun_sampling(sample_y):
            sample, y = sample_y
            if y == 0:
                parameters = variableMRFParameters.value.parameters
            else:
                parameters = variableMRFParameters.value.parametersForPY1
            return _sampling(sample, parameters)

        return rdd_x_y.map(myfun_sampling)

    def _runGibbsSampling_fromPY0(self, model, nbGibbsIter=1):
        constantMRFParameters = self.constantMRFParameters
        variableMRFParameters = self.variableMRFParameters

        gibbsMaxNbModalities = model.gibbsMaxNbModalities
        if gibbsMaxNbModalities > 1:

            def myfun_sampling_from_p_y0(sample):
                # new_sample = gibbsOneSampleFromPY0(
                new_sample = blockedGibbsSampler_PY0(
                    constantMRFParameters.value.explosedDisplayWeights,
                    constantMRFParameters.value.modalitiesByVarId,
                    variableMRFParameters.value.parameters,
                    sample,
                    nbGibbsIter,
                    gibbsMaxNbModalities,
                )
                return new_sample

        else:

            def myfun_sampling_from_p_y0(sample):
                new_sample = gibbsOneSampleFromPY0(
                    constantMRFParameters.value.explosedDisplayWeights,
                    constantMRFParameters.value.modalitiesByVarId,
                    variableMRFParameters.value.parameters,
                    sample,
                    nbGibbsIter,
                )
                return new_sample

        return self.rddSamples.map(myfun_sampling_from_p_y0)

    def UpdateSampleWeights(self, model):
        # Invalidating prediction because samples have changed
        self.prediction = None

    def _compute_prediction(self, model, rdd_x_edisplay_eclick):
        rdd_one_hot_index = self._convert_to_one_hot_index(model, rdd_x_edisplay_eclick)
        rdd_exploded = self._explode_one_hot_index(rdd_one_hot_index)
        projections = self._compute_prediction_projections(rdd_exploded).collectAsMap()
        z_on_z0 = projections[-1]
        del projections[-1]
        predict = np.zeros(model.parameters.size)
        indices = np.array(list(projections.keys()))
        values = np.array(list(projections.values()))
        predict[indices] = values
        nbsamplesInAggdata = np.exp(model.muIntercept) * (1 + np.exp(model.lambdaIntercept))
        predict *= nbsamplesInAggdata / z_on_z0
        return predict

    def PredictInternal(self, model):
        # here we only need to broadcast model parameters and collect all broadcast
        if self.variableMRFParameters is not None:
            self.broadcast_history.append(self.variableMRFParameters)
        self.variableMRFParameters = self.sparkSession.sparkContext.broadcast(VariableMRFParameters(model))
        # Invalidating prediction because parameters have changed
        self.prediction = None

    def GetPrediction(self, model):
        if self.prediction is None:
            rdd_sample_expdotproducts = self._compute_rdd_expdotproducts(model)
            rdd_x_edisplay_eclick = self._compute_edisplay_eclick(model, rdd_sample_expdotproducts)
            self.prediction = self._compute_prediction(model, rdd_x_edisplay_eclick)
        return self.prediction

    def _compute_rdd_expdotproducts(self, model):
        """
        Input: RDD containing tuple x
            x: vector with F features

        Output:  RDD containing tuple x,w,explambda
            x: vector with F features
            explambda: expo of dotproducts for clicks parameters
        """
        constantMRFParameters = self.constantMRFParameters
        variableMRFParameters = self.variableMRFParameters

        def expdotproducts(x):
            lambdas = 0
            for w in constantMRFParameters.value.clickWeights.values():
                lambdas += variableMRFParameters.value.parameters[w.feature.Values_(x) + w.offset]
            explambda = np.exp(lambdas + constantMRFParameters.value.lambdaIntercept)
            return x, explambda

        return self.rddSamples.map(expdotproducts)

    def _compute_edisplay_eclick(self, model, rdd_sample_expdotproducts):
        """
        Input: RDD containing tuple x,explambda
            x: vector with F features
            explambda: expo of dotproducts for clicks parameters

        Output: RDD containing tuple x,edisplay,eclick
            x: vector with F features
            edisplay: sample expectation of display
            eclick: sample expectation of display | click
        """
        # x, expmu, explambda
        constantMRFParameters = self.constantMRFParameters

        def _computePDisplays(x_explambda):
            x, explambda = x_explambda
            return x, 1, explambda / (1 + explambda)

        def _computePDisplaysFromPY0(x_explambda):
            x, explambda = x_explambda
            return x, 1 + explambda, explambda

        if self.sampleFromPY0:
            return rdd_sample_expdotproducts.map(_computePDisplaysFromPY0)
        else:
            return rdd_sample_expdotproducts.map(_computePDisplays)

    def _convert_to_one_hot_index(self, model, rdd_x_edisplay_eclick):
        """
        Input: RDD containing tuple x,edisplay,eclick,zi
            x: vector with F features
            edisplay: sample expectation of display
            eclick: sample expectation of display | click
            zi: Normalisation term used to compute final probability

        Output: RDD containing modality_enoclick_eclick_zi
            indices: vector of features and crossfeatures indices of x
            probability: probability weight of display or click
            zi: Normalisation term used to compute final probability
        """
        constantMRFParameters = self.constantMRFParameters

        def convertForFlatMap(x_edisplay_eclick):
            x, edisplay, eclick = x_edisplay_eclick
            proj_display = oneHotEncode(x, constantMRFParameters.value.explosedDisplayWeights)
            proj_click = oneHotEncode(x, constantMRFParameters.value.explosedClickWeights)
            return (
                (proj_display, edisplay),
                (proj_click, eclick),
                ([-1], edisplay),  # Keeping index '-1' to get sums of edisplays for normalization
            )

        return rdd_x_edisplay_eclick.flatMap(convertForFlatMap)

    def _explode_one_hot_index(self, rdd_one_hot_index):
        """
        Input: RDD containing vector indices,probability,zi
            indices: one-hot modalities indices
            probability: probability weight of display or click
            zi: Normalisation term used to compute final probability

        Output: RDD containing exploded modality_probability_zi
            modality: modality for which the expectation are computed
            probability: probability weight of display or click modality
            zi: Normalisation term used to compute final probability
        """

        def explodeProjections(modalities_value):
            modalities, value = modalities_value
            return zip(modalities, itertools.repeat(value))

        return rdd_one_hot_index.flatMap(explodeProjections)

    def _compute_prediction_projections(self, rdd_exploded):
        """
        Input: RDD containing exploded modality_probability_zi
            modality: modality for which the expectation are computed
            probability: probability weight of display or click modality
            zi: Normalisation term used to compute final probability

        Output: RDD grouped by modality
            modality: modality for which the expectation are computed
            probability: sum of probability weights for the modality
            zi: Partial sum of the normalisation term
        """
        return rdd_exploded.reduceByKey(operator.add)
