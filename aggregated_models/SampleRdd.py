import pandas as pd
import numpy as np
import random
import bisect
from collections import Counter
from aggregated_models.featuremappings import (
    CrossFeaturesMapping,
    FeatureMapping,
    DataProjection,
)
from aggregated_models.mrf_helpers import gibbsOneSampleFromPY0


MAXMODALITIES = 1e7


# set of samples of 'x' used internally by AggMRFModel
class SampleRdd:
    def __init__(
        self,
        projections,
        nbSamples=None,
        decollapseGibbs=False,
        sampleFromPY0=False,
        data=None,
    ):
        self.projections = projections
        self.decollapseGibbs = decollapseGibbs
        self.sampleFromPY0 = sampleFromPY0
        self.features = [p.feature for p in projections]
        self.featurenames = [f.Name for f in self.features]
        self.Size = nbSamples
        self.rddSamples = data
        self.allcrossmods = False
        self.prediction = None

    def get_rows(self):
        data = self.rddSamples.collect()
        return np.vstack([d for d in data])

    def UpdateSampleWithGibbs(self, model):
        self.rddSamples = self._updateSamplesWithGibbsRdd(model)
        self.rddSamples.checkpoint()
        self.rddSamples.count()

    def _updateSamplesWithGibbsRdd(self, model):
        constantMRFParameters = model.constantMRFParameters
        variableMRFParameters = model.variableMRFParameters

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
        rdd_sample_expdotproducts = self._compute_rdd_expdotproducts(model)
        rdd_sample_weights_expdotproducts = self._compute_weights(model, rdd_sample_expdotproducts)
        rdd_sample_enoclick_eclick_zi = self._compute_enoclick_eclick_zi(model, rdd_sample_weights_expdotproducts)
        self._compute_prediction(model, rdd_sample_enoclick_eclick_zi)

    def _compute_prediction(self, model, rdd_sample_enoclick_eclick_zi):
        rdd_one_hot_index = self._convert_to_one_hot_index(model, rdd_sample_enoclick_eclick_zi)
        rdd_exploded = self._explode_one_hot_index(rdd_one_hot_index)
        projections = self._compute_prediction_projections(rdd_exploded)
        projections = np.vstack(projections.collect())
        z_on_z0 = projections[:, 2].sum()
        predict = np.zeros(model.parameters.size)
        indices = projections[:, 0].astype(int)
        predict[indices] = (projections[:, 1]) * self.Size / z_on_z0
        self.prediction = predict

    def PredictInternal(self, model):
        rdd_sample_expdotproducts = self._compute_rdd_expdotproducts(model)
        rdd_sample_weights_expdotproducts = self._compute_weights(model, rdd_sample_expdotproducts)
        rdd_sample_enoclick_eclick_zi = self._compute_enoclick_eclick_zi(model, rdd_sample_weights_expdotproducts)
        self._compute_prediction(model, rdd_sample_enoclick_eclick_zi)

    def GetPrediction(self, model):
        # self._compute_prediction(model)
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
        constantMRFParameters = model.constantMRFParameters
        variableMRFParameters = model.variableMRFParameters

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
        constantMRFParameters = model.constantMRFParameters

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
        constantMRFParameters = model.constantMRFParameters

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

        constantMRFParameters = model.constantMRFParameters

        def convertForFlatMap(x_enoclick_eclick_zi):
            x, enoclick, eclick, zi = x_enoclick_eclick_zi
            nbDisplayWeights = len(constantMRFParameters.value.displayWeights)
            nbClickWeights = len(constantMRFParameters.value.clickWeights)
            proj_display = np.zeros((nbDisplayWeights + 2))
            proj_click = np.zeros((nbClickWeights + 2))
            proj_display[-2] = enoclick + eclick
            proj_click[-2] = eclick
            proj_display[-1] = zi  # broadcast zi only once per vector to avoid normalization error
            proj_click[-1] = 0  # broadcast zi only once per vector to avoid normalization error
            for k, w in enumerate(constantMRFParameters.value.displayWeights.values()):
                proj_display[k] = w.feature.Values_(x) + w.offset
            for k, w in enumerate(constantMRFParameters.value.clickWeights.values()):
                proj_click[k] = w.feature.Values_(x) + w.offset
            return [proj_display, proj_click]

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

        def explodeProjections(mod_proba_zi):
            explosion = np.zeros((len(mod_proba_zi) - 2, 3))
            explosion[:, 0] = mod_proba_zi[:-2]  # Explode modalities
            explosion[:, 1] = mod_proba_zi[-2]  # Set weights (1 per modality)
            explosion[0, 2] = mod_proba_zi[-1]  # Set normalization constant (once only per vector)
            return explosion

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
        return (
            rdd_exploded.keyBy(lambda sample_prediction: sample_prediction[0])
            .mapValues(lambda sample_prediction: sample_prediction[1:])
            .reduceByKey(lambda prediction, prediction_1: prediction + prediction_1)
            .map(lambda proj_prediction: np.hstack((proj_prediction[0], proj_prediction[1])))
        )
