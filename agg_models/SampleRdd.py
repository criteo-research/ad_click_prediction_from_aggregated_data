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
from agg_models.mrf_helpers import fastGibbsSampleFromPY0
import traceback


MAXMODALITIES = 1e7


# set of samples of 'x' used internally by AggMRFModel
class SampleRdd:
    def __init__(
        self,
        projections,
        nbSamples=None,
        decollapseGibbs=False,
        sampleFromPY0=False,
        maxNbRowsperGibbsUpdate=50,
        data=None,
    ):
        self.projections = projections
        self.decollapseGibbs = decollapseGibbs
        self.sampleFromPY0 = sampleFromPY0
        self.features = [p.feature for p in projections]
        self.featurenames = [f.Name for f in self.features]
        self.Size = nbSamples
        self.data = data
        self.allcrossmods = False
        self.prediction = None

    def UpdateSampleWithGibbs(self, model):
        #  print("UpdateSampleWithGibbs")
        self.data = self.updateSamplesWithGibbsRdd(model)
        self.data.checkpoint()

    def updateSamplesWithGibbsRdd(self, model):
        constantMRFParameters = model.constantMRFParameters
        variableMRFParameters = model.variableMRFParameters

        def myfun_sampling_from_p_y0(sampleWithWeight):
            sample = fastGibbsSampleFromPY0(
                constantMRFParameters.value.explosedDisplayWeights,
                constantMRFParameters.value.modalitiesByVarId,
                variableMRFParameters.value.parameters,
                sampleWithWeight[0],
                1,
            )
            return sample, sampleWithWeight[1]

        return self.data.map(myfun_sampling_from_p_y0)

    def UpdateSampleWeights(self, model):
        #  print("UpdateSampleWeights")
        rdd_sample_weights_with_expdotproducts = self.compute_rdd_expdotproducts(model)
        rdd_sample_updated_weights_with_expdotproducts = self.compute_weights(
            model, rdd_sample_weights_with_expdotproducts
        )
        self.data = self.compute_enoclick_eclick_zi(model, rdd_sample_updated_weights_with_expdotproducts)
        self.compute_prediction(model)

    def compute_prediction(self, model):
        #  print("compute_prediction")
        pdisplays, z_on_z0 = self.compute_prediction_reduce(model, self.data)
        # Compute z0_on_z : 1 / np.mean(z_i) = np.sum(z_zi) / nbSamples
        predict = pdisplays * self.Size / z_on_z0
        self.prediction = predict

    def PredictInternal(self, model):
        #  print("PredictInternal")
        rdd_sample_weights_with_expdotproducts = self.compute_rdd_expdotproducts(model)
        self.data = self.compute_enoclick_eclick_zi(model, rdd_sample_weights_with_expdotproducts)
        self.compute_prediction(model)

    def GetPrediction(self, model):
        # self.compute_prediction(model)
        return self.prediction

    def compute_rdd_expdotproducts(self, model):
        #  print("computedotprods")
        """
        Input: RDD containing tuple x,w,(...)
            x: matrix (K,F) of K samples with F features
            w: vector of K weights
            (...): potential leftover from previous iterations, ignored

        Output:  RDD containing tuple x,w,expmu,explambda
            x, w: unchanged
            expmu: exp of dotproducts for display parameters
            explambda: expo of dotproducts for clicks parameters
        """
        constantMRFParameters = model.constantMRFParameters
        variableMRFParameters = model.variableMRFParameters

        def expdotproducts(samples_weights):
            x, weights = samples_weights[0], samples_weights[1]
            t_x = x.transpose()
            mus = np.zeros(x.shape[0])
            lambdas = np.zeros(x.shape[0])
            for w in constantMRFParameters.value.displayWeights.values():
                mus += variableMRFParameters.value.parameters[w.feature.Values_(t_x) + w.offset]
            for w in constantMRFParameters.value.clickWeights.values():
                lambdas += variableMRFParameters.value.parameters[w.feature.Values_(t_x) + w.offset]
            expmu = np.exp(mus + constantMRFParameters.value.muIntercept)
            explambda = expmu * np.exp(lambdas + constantMRFParameters.value.lambdaIntercept)
            return x, weights, expmu, explambda

        return self.data.map(expdotproducts)

    def compute_weights(self, model, rdd_samples_weights_with_expdotproducts):
        """
        Input: RDD containing tuple x,w,expmu,explambda
            x: matrix (K,F) of K samples with F features
            w: vector of K weights
            expmu: exp of dotproducts for display parameters
            explambda: expo of dotproducts for clicks parameters

        Output: RDD containing tuple x,w,expmu,explambda
            x: unchanged
            w: udpated weights from prediction
            expmu: unchanged
            explambda: unchanged
        """
        #  print("computeProbaSamples")
        #  print("SetWeights")
        constantMRFParameters = model.constantMRFParameters

        def _computeWeightFromPY0(samples_weights_exp_mu_lambda):
            x, weights, expmu, explambda = samples_weights_exp_mu_lambda
            weights = constantMRFParameters.value.norm / expmu / constantMRFParameters.value.nbSamples
            return x, weights, expmu, explambda

        def _computeWeight(samples_weights_exp_mu_lambda):
            x, weights, expmu, explambda = samples_weights_exp_mu_lambda
            weights = constantMRFParameters.value.norm / (expmu + explambda) / constantMRFParameters.value.nbSamples
            return x, weights, expmu, explambda

        if self.sampleFromPY0:
            return rdd_samples_weights_with_expdotproducts.map(_computeWeightFromPY0)
        else:
            return rdd_samples_weights_with_expdotproducts.map(_computeWeight)

    def compute_enoclick_eclick_zi(self, model, rdd_samples_weights_with_expdotproducts):
        """
        Input: RDD containing tuple x,w,expmu,explambda
            x: matrix (K,F) of K samples with F features
            w: vector of K weights
            expmu: exp of dotproducts for display parameters
            explambda: expo of dotproducts for clicks parameters

        Output: RDD containing tuple x,w,enoclick,eclick,z_i
            x, w: unchanged
            enoclick: sample expectation of display | no click
            eclick: sample expectation of display | click
            z_i: Term used to compute P(Y)
        """
        #  print("compute_enoclick_eclick")
        # x, expmu, explambda
        constantMRFParameters = model.constantMRFParameters

        def _computePDisplays(samples_weight_mu_lambda):
            x, weights, expmu, explambda = samples_weight_mu_lambda
            enoclick = expmu * weights
            eclick = explambda * weights
            z_i = (1 + explambda / expmu).sum()
            if constantMRFParameters.value.sampleFromPY0:  # correct importance weigthing formula
                eclick *= 1 + np.exp(constantMRFParameters.value.lambdaIntercept)
                enoclick *= 1 + np.exp(constantMRFParameters.value.lambdaIntercept)
            return x, weights, enoclick, eclick, z_i

        return rdd_samples_weights_with_expdotproducts.map(_computePDisplays)

    def compute_prediction_reduce(self, model, x_w_enoclick_eclick):
        """
        Input: RDD containing tuple x,w,enoclick,eclick,z_i
            x: matrix (K,F) of K samples with F features
            w: vector of K weights
            expmu: exp of dotproducts for display parameters
            explambda: expo of dotproducts for clicks parameters

        Output: Tuple containing p,1/z_0
            p: np.array, pdisplay
            1/z_0: float used to compute P(Y)
        """

        constantMRFParameters = model.constantMRFParameters

        def computePredictions(samples_w_enoclick_eclick_zi):
            p = np.zeros(constantMRFParameters.value.nbParameters)
            x, w, enoclick, eclick, z_i = samples_w_enoclick_eclick_zi
            t_x = x.transpose()
            for w in constantMRFParameters.value.displayWeights.values():
                p[w.indices] = w.feature.Project_(t_x, enoclick + eclick)  # Correct for grads
            for w in constantMRFParameters.value.clickWeights.values():
                p[w.indices] = w.feature.Project_(t_x, eclick)
            return p, z_i

        return x_w_enoclick_eclick.map(computePredictions).reduce(
            lambda p_z, p_z1: (p_z[0] + p_z1[0], p_z[1] + p_z1[1])
        )
