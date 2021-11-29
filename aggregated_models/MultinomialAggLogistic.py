import numpy as np
from aggregated_models.CrossFeaturesSet import *
from aggregated_models import Optimizers
from aggregated_models.baseaggmodel import BaseAggModel
from aggregated_models.mrf_helpers import *
from joblib import Parallel, delayed
from aggregated_models.aggdataset import *


class MultinomialAggLogistic(BaseAggModel):
    def __init__(self, aggdata, samples, features, label, regulL2=1.0):
        aggdata.aggDisplays[label].feature._fid = len(features)
        super().__init__(aggdata, features)

        self.samples = samples
        if not hasattr(self.samples, "__len__"):
            if len(features) > 1:
                raise Exception("Must pass samples matrix if there are several features")
            print(f"Generating {samples} samples of feature {features[0]}")
            p = aggdata.aggDisplays[features[0]].Data
            x = np.array([weightedSampleNUMBA(p) for _ in range(0, samples)])
            self.samples = np.array(x)
        if len(self.samples.shape) == 1:  # case when only one feature
            self.samples = self.samples.reshape((len(self.samples), 1))
            print(f"samples reshaped to {self.samples.shape}")

        self.nbsamples = len(self.samples)
        self.regulL2 = regulL2
        self.label = label
        cfsString = "*&" + label
        self.cfs = [self.label] + parseCFNames(self.features, cfsString)
        self.projections = {var: self.aggdata.aggDisplays[var] for var in self.cfs}
        self.weights, self.nbCoefs = self.prepareWeights(self.cfs)
        self.parameters = np.zeros(self.nbCoefs)
        self.Data = self.getAggDataVector(self.weights, self.projections)
        w0 = self.weights[self.label]
        ind = w0.indices
        self.parameters[ind] = np.log(10 + self.Data[ind])
        self.parameters[ind] -= self.parameters[ind].mean()

        labelEncoding = self.projections[self.label].feature
        self.modalities = np.arange(0, labelEncoding.Size)

        self.globalRescaling = self.aggdata.Nbdisplays / self.nbsamples
        self.setRescalingRatios()
        self.nbIters = 0
        self.alpha = 0.5 / len(self.features)

        print(f"Multinomial {self.features}->{self.label} , nbsamples:{self.nbsamples}")

    def exportWeights(self):
        n = len(self.weights)
        coefsv = np.zeros(n, dtype=np.int32)
        coefsv2 = np.zeros(n, dtype=np.int32)
        offsets = np.zeros(n, dtype=np.int32)
        otherfeatureid = np.zeros(n, dtype=np.int32)
        modulos = np.zeros(n, dtype=np.int32)
        for i, w in enumerate(self.weights.values()):
            offsets[i] = w.offset
            modulos[i] = w.feature.Size
            coefsv[i] = 1
            coefsv2[i] = 0
            otherfeatureid[i] = 0
            if type(w.feature) is CrossFeatureEncoding:
                if self.label == w.feature._v1:
                    coefsv2[i] = w.feature.coefV2
                    otherfeatureid[i] += self.features.index(w.feature._v2)
                    # formula for indice of the crossmodality (V1=v1,V2=v2) in the parameter vector is :
                    #  w.offset + v1 * coefsv + v2 * coefsv2
                else:
                    coefsv[i] = w.feature.coefV2
                    coefsv2[i] = 1
                    otherfeatureid[i] = self.features.index(w.feature._v1)

        return coefsv, coefsv2, offsets, otherfeatureid, modulos

    def computePdata(self):
        pdata = self.computePdataInternal()
        pdata *= self.globalRescaling
        self.pData = pdata
        return pdata

    @staticmethod
    def splitInSlices(x, maxNbRows):
        starts = np.arange(0, len(x), maxNbRows)
        slices = [(x[start : start + maxNbRows]) for start in starts]
        return slices

    def computePdataInternal(self, para=True):
        coefsv, coefsv2, offsets, otherfeatureid, modulos = self.exportWeights()
        modalities = self.modalities
        paramsVector = self.parameters

        if not para:
            return ExpectedK(self.samples, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)

        def computeExpectedK(x):
            return ExpectedK(x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)

        slices = MultinomialAggLogistic.splitInSlices(self.samples, 5000)
        runner = Parallel(n_jobs=14)
        jobs = [delayed(computeExpectedK)(myslice) for myslice in slices]
        samplesSlices = runner(jobs)
        return np.sum(samplesSlices, axis=0)

    def sampleY(self):
        coefsv, coefsv2, offsets, otherfeatureid, modulos = self.exportWeights()
        modalities = self.modalities
        paramsVector = self.parameters
        y = SampleFromKernel(self.samples, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)
        return y

    def finalize(self):
        y = self.sampleY()
        return np.c_[self.samples[:, : len(self.features)], y]

    def computeGradient(self):  # grad of loss
        preds = self.computePdata()
        gradient = -self.Data + preds
        # gradient = -self.Data * np.minimum(1, self.rescalingRatio) + preds / np.maximum(1, self.rescalingRatio)
        gradient += 2 * self.parameters * self.regulL2
        return gradient

    def computeInvHessianDiagAtOptimum(self):  # grad of loss
        return 1 / (self.regulL2 * 2 + self.Data)

    def computeInvHessianDiag(self, alpha=0.9):  # grad of loss
        preds = self.pData
        preds = preds * alpha + self.Data * (1 - alpha)  # averaging with value at optimum
        return 1 / (self.regulL2 * 2 + preds)

    def setRescalingRatios(self):
        rescalingRatios = []
        offsetsInRescalingRatio = []
        for f in self.features:
            offsetsInRescalingRatio += [len(rescalingRatios)]
            proj = self.aggdata.aggDisplays[f]
            countsOnSamples = proj.feature.Project_(self.samples.transpose(), np.ones(self.nbsamples))
            countsOnSamples *= self.globalRescaling
            rescalingRatios += [x for x in (countsOnSamples + 1) / (proj.Data + 1)]
        self.rescalingRatios = np.array(rescalingRatios)
        self.offsetsInRescalingRatio = np.array(offsetsInRescalingRatio, dtype=np.int32)

    def fit(self, nbIter=50, verbose=False):
        def endIterCallback():
            self.nbIters += 1

        Optimizers.simpleGradientStep(self, nbIter, self.alpha, endIterCallback)
        # Optimizers.lbfgs(self, nbiter=nbIter, alpha=0.01, verbose=verbose)

    def update(self):
        pass  # for Optimizers.simpleGradientStep

    def apply(self, x):
        SampleOneFromKernel(x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)
