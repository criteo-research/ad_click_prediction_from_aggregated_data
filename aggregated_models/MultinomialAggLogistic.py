import numpy as np
from aggregated_models.CrossFeaturesSet import *
from aggregated_models import Optimizers
from aggregated_models.baseaggmodel import BaseAggModel
from aggregated_models.mrf_helpers import *
from joblib import Parallel, delayed
from aggregated_models.aggdataset import *
import sys
import os
from aggregated_models.agg_mrf_model import *
from aggregated_models.aggLogistic import AggLogistic


class MultinomialAggLogistic(BaseAggModel):
    def __init__(self, aggdata, samples, features, label, regulL2=1.0, ss=None, nbPartitions=1000, rescaling=False):
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
        self.rescaling = rescaling
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
        self.nbIters = 0
        self.alpha = 0.5 / len(self.features)

        print(f"Multinomial {self.features}->{self.label} , nbsamples:{self.nbsamples}")

        self.ss = ss
        if self.ss is not None:
            self.rdd, self.rdds = MultinomialAggLogistic.parallelize(self.samples, ss, nbPartitions)
            # self.rdd = ss.sparkContext.parallelize( self.samples , nbPartitions ).cache()
            self.buildConstantBroadcast()

        self.globalRescaling = self.aggdata.Nbdisplays / self.nbsamples
        self.setRescalingRatios()

    def buildConstantBroadcast(self):
        coefsv, coefsv2, offsets, otherfeatureid, modulos = self.exportWeights()
        modalities = self.modalities
        tobroadcast = (coefsv, coefsv2, offsets, otherfeatureid, modulos, modalities)
        self.constantBroadcast = self.ss.sparkContext.broadcast(tobroadcast)

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

    @staticmethod
    def parallelize(x, ss, totalnbpartitions):
        nbslices = sys.getsizeof(x) // (256_000_112 // 10)  # crashed when the data size was around 256Mo

        if nbslices < 1:
            nbslices = 1
        slices = MultinomialAggLogistic.splitInSlices(x, x.shape[0] // nbslices)

        nbpartitions = totalnbpartitions // nbslices
        if nbpartitions < 1:
            nbpartitions = 1
        fullrdd = None
        rdds = []
        for s in slices:
            rdd = ss.sparkContext.parallelize(s, nbpartitions).cache()
            rdd.count()
            rdds.append(rdd)
            if fullrdd is None:
                fullrdd = rdd
            else:
                fullrdd = fullrdd.union(rdd)

        return fullrdd, rdds

    def pysparkComputePdataInternal(self):
        paramBroadcast = self.ss.sparkContext.broadcast(self.parameters)
        constantBroadcast = self.constantBroadcast

        def tomapOnPartitions(iterator):
            try:
                x = np.stack(iterator)
            except:
                return  # happens on empty partitions
            coefsv, coefsv2, offsets, otherfeatureid, modulos, modalities = constantBroadcast.value
            paramsVector = paramBroadcast.value
            result = ExpectedK(x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)
            yield result

        sums = self.rdd.mapPartitions(tomapOnPartitions)

        nbMos = sys.getsizeof(self.parameters) * self.rdd.getNumPartitions() / 1000 / 1000
        if nbMos > 1000:
            result = sums.treeReduce(np.add, depth=3)
        elif nbMos > 100:
            result = sums.treeReduce(np.add, depth=2)
        else:
            result = sums.reduce(np.add)
        try:
            paramBroadcast.destroy()
        except:
            print("failed:: paramBroadcast.destroy()")
        return result

    def computePdataInternal(self, para=True):

        if self.ss is not None:
            return self.pysparkComputePdataInternal()

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
        if self.ss is not None:
            return self.pysparkfinalize()
        y = self.sampleY()
        return np.c_[self.samples[:, : len(self.features)], y]

    def pysparkfinalize(self):
        paramBroadcast = self.ss.sparkContext.broadcast(self.parameters)
        constantBroadcast = self.constantBroadcast

        def tomapOnPartitions(iterator):
            coefsv, coefsv2, offsets, otherfeatureid, modulos, modalities = constantBroadcast.value
            paramsVector = paramBroadcast.value
            try:
                x = np.stack(iterator)
            except:
                return  # happens on empty partitions
            # compile numba
            y = SampleFromKernel(x[:2, :], coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)
            # really run computation
            y = SampleFromKernel(x, coefsv, coefsv2, offsets, modulos, otherfeatureid, modalities, paramsVector)
            # x = np.c_[x,y]
            yield y

        y = self.rdd.mapPartitions(tomapOnPartitions).collect()
        # a = []
        # for fid in range( len(self.features  )+1):
        #    a.append (np.array (result.map(lambda x : x[fid]).collect())) # collect all at once => crashing spark
        # a = np.vstack(a).transpose()

        try:
            paramBroadcast.destroy()
        except:
            print("failed:: paramBroadcast.destroy()")
        # result.unpersist()
        y = np.hstack(y)
        return np.c_[self.samples, y]

    def setRescalingRatios(self):
        if True:
            from matplotlib import pyplot as plt

            preds = self.computePdata()
            w = list(self.weights.values())[-1]
            encoding = w.feature
            phi_x_samples = encoding.marginalize(preds[w.indices], self.label)
            phi_x_train = encoding.marginalize(self.Data[w.indices], self.label)
            ratio = (phi_x_samples + 1) / (phi_x_train + 1)
            plt.figure()
            plt.plot(ratio)
            plt.title("Rescaling : " + str(w))
            plt.show()
        if self.rescaling:
            self.rescalingRatio = self.computeRescalingRatio()
        else:
            self.rescalingRatio = np.ones(len(self.parameters))

    def computeRescalingRatio(self):
        preds = self.computePdata()
        ratio = np.ones(len(preds))
        for w in self.weights.values():
            if not "CrossFeatureEncoding" in str(type(w.feature)):
                continue
            ratio[w.indices] = self.cfRatio(preds, w)
        return ratio

    def cfRatio(self, preds, w):
        encoding = w.feature
        phi_x_samples = encoding.marginalize(preds[w.indices], self.label)
        phi_x_train = encoding.marginalize(self.Data[w.indices], self.label)
        ratio_f1 = (phi_x_samples + 1) / (phi_x_train + 1)
        values = encoding.modalitiesOtherFeature(self.label)
        return ratio_f1[values]

    def computeGradient(self):  # grad of loss
        preds = self.computePdata()
        gradient = -self.Data + preds
        gradient = -self.Data * np.minimum(1, self.rescalingRatio) + preds / np.maximum(1, self.rescalingRatio)
        gradient += 2 * self.parameters * self.regulL2
        return gradient

    def computeInvHessianDiagAtOptimum(self):  # grad of loss
        return 1 / (self.regulL2 * 2 + self.Data)

    def computeInvHessianDiag(self, alpha=0.9):  # grad of loss
        preds = self.pData
        preds = preds * alpha + self.Data * (1 - alpha)  # averaging with value at optimum
        return 1 / (self.regulL2 * 2 + preds)

    def setRescalingRatiosOLD(self):
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

    def clean(self):
        if self.ss is not None:
            try:
                self.constantBroadcast.destroy()
            except:
                print("failed:: constantBroadcast.destroy()")
            for rdd in self.rdds:
                try:
                    rdd.unpersist()
                except:
                    print("failed:: rdd.unpersist()")


class LogisticOnMultinomialSamples:
    def __init__(
        self,
        aggdata: AggDataset,
        config: AggMRFModelParams,
        ss: Optional[SparkSession] = None,
        nbIters=100,
        savename=None,
    ):
        if config.maxNbRowsPerSlice == 25:
            config.maxNbRowsPerSlice = 500  ## More raisonable default value

        self.aggdata = aggdata
        self.config = config
        self.ss = ss
        self.nbIters = nbIters

        if savename is None:
            savename = f"Multinomials_{np.random.randint(10000000)}"
        self.savename = savename
        self.features = self.config.features

        self.nbPartitions = max(1, self.config.nbSamples // self.config.maxNbRowsPerSlice)
        print(f"nbPartitions = {self.nbPartitions}")

        if os.path.isfile(self.savename + ".npy"):
            self.samples = np.load(self.savename + ".npy")
            print(f"Found samples with {self.nbFeaturesInSamples()} features  shape={self.samples.shape}")
        else:
            self.samples = None
        self.modelParams = {}

    def nbFeaturesInSamples(self):
        if self.samples is None:
            return 1
        return self.samples.shape[1]

    def fit(self):
        while self.nbFeaturesInSamples() < len(self.features):
            self.fitMultinomial()

        self.agglogistic = AggLogistic(
            self.aggdata, self.features, clicksCfs=self.config.clicksCfs, regulL2=self.config.regulL2Click
        )
        alpha = 0.5 / len(self.agglogistic.clickWeights)
        self.agglogistic.fit(self.samples.transpose(), nbIter=self.nbIters, alpha=alpha)

    def predictDF(self, df, name="p"):
        return self.agglogistic.predictDF(df, name)

    def fitMultinomial(self):
        nbf = self.nbFeaturesInSamples()
        samples = self.samples
        if samples is None:
            samples = self.config.nbSamples
        label = self.features[nbf]
        self.multinomial = MultinomialAggLogistic(
            self.aggdata,
            samples,
            self.features[:nbf],
            label,
            ss=self.ss,
            nbPartitions=self.nbPartitions,
            rescaling=self.config.multinomialRescaling,
        )
        print(f"Starting fit of { label } from {nbf} previous features")
        self.multinomial.fit(self.nbIters)
        x = self.multinomial.finalize()
        self.samples = x
        np.save(self.savename, self.samples)
        self.multinomial.clean()
        if self.ss is not None:
            self.ss.catalog.clearCache()
