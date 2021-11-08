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


# set of samples of 'x' used internally by AggMRFModel
class SampleRdd:
    def __init__(
        self,
        projections,
        sparkSession,
        constantModelParameters: ConstantMRFParameters = None,
        variableModelParameters: VariableMRFParameters = None,
        nbSamples=100,
        someObsoleteParam=False,
        sampleFromPY0=False,
        maxNbRowsPerSlice=500,
        data=None,
    ):
        self.projections = projections
        self.sampleFromPY0 = sampleFromPY0
        self.Size = nbSamples
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

    @property
    def rdd(self):
        return self.rddSamples

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
        # Invalidating prediction because samples have changed
        self.prediction = None

    def cleanBroadcasts(self):
        for k in self.broadcast_history:
            # after the checkpoint, we can safely delete old broadcasted parameters
            try:
                k.destroy()  # sometimes crashing with a "timeout". seems to happen when one of the nodes of the sesssion is having issues (memory or other ?)
            except:
                print("Failure while trying to destroy a broadcast")
        self.broadcast_history = list()

    def _runGibbsSampling(self, model, nbGibbsIter=1):
        return model.pysparkGibbsSampler(self, nbGibbsIter)

    def _rebuid_predictions(self, model, projections):
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
            self.prediction = model.pysparkPredict(self)
        return self.prediction / model.aggdata.Nbdisplays
