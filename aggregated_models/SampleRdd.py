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
        maxNbRowsPerSlice=50,
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

    def get_rows(self) -> np.ndarray:
        data = self.rddSamples.collect()
        return np.array([d for d in data])

    def UpdateSampleWithGibbs(self, model):
        rddnew = self._updateSamplesWithGibbsRdd(model)
        rddnew.cache()  # caching before checkpoint, because checkpoint would trigger the computation twice.
        rddnew.checkpoint()
        rddnew.count()  # force compute before uncache
        self.rddSamples.unpersist()  # clean previous rdd
        self.rddSamples = rddnew
        for k in self.broadcast_history:
            # after the checkpoint, we can safely delete old broadcasted parameters
            k.destroy()
        self.broadcast_history = list()

    def _updateSamplesWithGibbsRdd(self, model):
        constantMRFParameters = self.constantMRFParameters
        variableMRFParameters = self.variableMRFParameters

        def myfun_sampling_from_p_y0(sample):
            new_sample = gibbsOneSampleFromPY0(
                constantMRFParameters.value.explosedDisplayWeights,
                constantMRFParameters.value.modalitiesByVarId,
                variableMRFParameters.value.parameters,
                sample,
                1,
            )
            return new_sample

        return self.rddSamples.map(myfun_sampling_from_p_y0)

    def UpdateSampleWeights(self, model):
        # Invalidating prediction because samples have changed
        self.prediction = None

    def _compute_prediction(self, model, rdd_sample_enoclick_eclick_zi):
        rdd_one_hot_index = self._convert_to_one_hot_index(model, rdd_sample_enoclick_eclick_zi)
        rdd_exploded = self._explode_one_hot_index(rdd_one_hot_index)
        projections = self._compute_prediction_projections(rdd_exploded).collectAsMap()
        z_on_z0 = projections[-1]
        del projections[-1]
        predict = np.zeros(model.parameters.size)
        indices = np.array(list(projections.keys()))
        values = np.array(list(projections.values()))
        predict[indices] = values * self.Size / z_on_z0
        return predict

    def PredictInternal(self, model):
        # here we only need to broadcast model parameters and collect all broadcast
        if self.variableMRFParameters is not None:
            self.broadcast_history.append(self.variableMRFParameters)
        self.variableMRFParameters = self.sparkSession.sparkContext.broadcast(VariableMRFParameters(model.parameters))
        # Invalidating prediction because parameters have changed
        self.prediction = None

    def GetPrediction(self, model):
        if self.prediction is None:
            rdd_sample_expdotproducts = self._compute_rdd_expdotproducts(model)
            rdd_sample_weights_expdotproducts = self._compute_weights(model, rdd_sample_expdotproducts)
            rdd_sample_enoclick_eclick_zi = self._compute_enoclick_eclick_zi(model, rdd_sample_weights_expdotproducts)
            self.prediction = self._compute_prediction(model, rdd_sample_enoclick_eclick_zi)
        return self.prediction

    def _compute_rdd_expdotproducts(self, model):
        """
        Input: RDD containing tuple x
            x: vector with F features

        Output:  RDD containing tuple x,w,expmu,explambda
            x: vector with F features
            expmu: exp of dotproducts for display parameters
            explambda: expo of dotproducts for clicks parameters
        """
        constantMRFParameters = self.constantMRFParameters
        variableMRFParameters = self.variableMRFParameters

        def expdotproducts(x):
            mus = 0
            lambdas = 0
            for w in constantMRFParameters.value.displayWeights.values():
                mus += variableMRFParameters.value.parameters[w.feature.Values_(x) + w.offset]
            for w in constantMRFParameters.value.clickWeights.values():
                lambdas += variableMRFParameters.value.parameters[w.feature.Values_(x) + w.offset]
            expmu = np.exp(mus + constantMRFParameters.value.muIntercept)
            explambda = expmu * np.exp(lambdas + constantMRFParameters.value.lambdaIntercept)
            return x, expmu, explambda

        return self.rddSamples.map(expdotproducts)

    def _compute_weights(self, model, rdd_x_expdotproducts):
        """
        Input: RDD containing tuple x,w,expmu,explambda
            x: vector with F features
            w: vector of K weights
            expmu: exp of dotproducts for display parameters
            explambda: exp of dotproducts for clicks parameters

        Output: RDD containing tuple x,w,expmu,explambda
            x: vector with F features
            w: udpated weights from prediction
            expmu: exp of dotproducts for display parameters
            explambda: exp of dotproducts for clicks parameters
        """
        constantMRFParameters = self.constantMRFParameters

        def _computeWeightFromPY0(x_exp_mu_lambda):
            x, expmu, explambda = x_exp_mu_lambda
            weights = constantMRFParameters.value.norm / expmu / constantMRFParameters.value.nbSamples
            return x, weights, expmu, explambda

        def _computeWeight(x_exp_mu_lambda):
            x, expmu, explambda = x_exp_mu_lambda
            weights = constantMRFParameters.value.norm / (expmu + explambda) / constantMRFParameters.value.nbSamples
            return x, weights, expmu, explambda

        if self.sampleFromPY0:
            return rdd_x_expdotproducts.map(_computeWeightFromPY0)
        else:
            return rdd_x_expdotproducts.map(_computeWeight)

    def _compute_enoclick_eclick_zi(self, model, rdd_x_weights_expdotproducts):
        """
        Input: RDD containing tuple x,w,expmu,explambda
            x: vector with F features
            w: weight
            expmu: exp of dotproducts for display parameters
            explambda: expo of dotproducts for clicks parameters

        Output: RDD containing tuple x,w,enoclick,eclick,zi
            x: vector with F features
            enoclick: sample expectation of display | no click
            eclick: sample expectation of display | click
            zi: Term used to compute P(Y)
        """
        # x, expmu, explambda
        constantMRFParameters = self.constantMRFParameters

        def _computePDisplays(x_weight_mu_lambda):
            x, weights, expmu, explambda = x_weight_mu_lambda
            enoclick = expmu * weights
            eclick = explambda * weights
            zi = (1 + explambda / expmu).sum()
            if constantMRFParameters.value.sampleFromPY0:  # correct importance weigthing formula
                eclick *= 1 + np.exp(constantMRFParameters.value.lambdaIntercept)
                enoclick *= 1 + np.exp(constantMRFParameters.value.lambdaIntercept)
            return x, enoclick, eclick, zi

        return rdd_x_weights_expdotproducts.map(_computePDisplays)

    def _convert_to_one_hot_index(self, model, rdd_x_enoclick_eclick_zi):
        """
        Input: RDD containing tuple x,enoclick,eclick,zi
            x: vector with F features
            enoclick: sample expectation of display | no click
            eclick: sample expectation of display | click
            zi: Normalisation term used to compute final probability

        Output: RDD containing modality_enoclick_eclick_zi
            indices: vector of features and crossfeatures indices of x
            probability: probability weight of display or click
            zi: Normalisation term used to compute final probability
        """
        constantMRFParameters = self.constantMRFParameters

        def convertForFlatMap(x_enoclick_eclick_zi):
            x, enoclick, eclick, zi = x_enoclick_eclick_zi
            proj_display = oneHotEncode(x, constantMRFParameters.value.explosedDisplayWeights)
            proj_click = oneHotEncode(x, constantMRFParameters.value.explosedClickWeights)
            return (
                (proj_display, enoclick + eclick),
                (proj_click, eclick),
                ([-1], zi),  # Keeping index '-1' to get sums of zi
            )

        return rdd_x_enoclick_eclick_zi.flatMap(convertForFlatMap)

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
