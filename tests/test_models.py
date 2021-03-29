from pytest import approx, fixture
# helpers to compute metrics
from aggregated_models.validation import MetricsComputer

# baselines
from aggregated_models.basicmodels import LogisticModel, NaiveBayesModel, LogisticModelWithCF
from aggregated_models.aggLogistic import AggLogistic

# code to prepare the aggregated dataset
from aggregated_models.featuremappings import AggDataset

## Most relevant code is there:
from aggregated_models.agg_mrf_model import AggMRFModel, fastGibbsSample, fastGibbsSampleFromPY0
import aggregated_models.agg_mrf_model
import pandas as pd


class ModelTestData():
    def __init__(self):
        self.train = pd.read_parquet("./tests/resources/small_train.parquet")
        self.valid = pd.read_parquet("./tests/resources/small_valid.parquet")

        self.features = ["cat1", "cat4", "cat6", "cat8", "cat9"]
        self.label = "click"

        self.validator = MetricsComputer(self.label)

        # parameters for of the privacy protecting noise.
        self.epsilon = None  # Set to None to get no noise.
        self.delta = None
        self.aggdata = AggDataset(
            self.features, "*&*", self.train, self.label, self.epsilon, self.delta, maxNbModalities=10000
        )


@fixture
def model_data():
    return ModelTestData()


def test_agg_data(model_data):
    assert model_data.aggdata.label == "click"
    assert len(model_data.aggdata.aggDisplays) == 15  # Number of queries for N features = N(N+1)/2


def test_logistic(model_data):
    regulL2 = 16
    logisticCfs = LogisticModelWithCF(
        model_data.label, model_data.features, "*&*", model_data.train, hashspace=2 ** 22, lambdaL2=regulL2
    )
    logisticCfs.fit(model_data.train)
    assert model_data.validator.getLLH(logisticCfs, model_data.train) == approx(0.060, rel=0.1)
    assert model_data.validator.getLLH(logisticCfs, model_data.valid) == approx(0.060, rel=0.1)


def test_agg_logistic(model_data):
    regulL2 = 16
    aggLogistic = AggLogistic(model_data.aggdata, model_data.features, clicksCfs="*&*", regulL2=regulL2)
    aggLogistic.fit(model_data.train[model_data.features], nbIter=200)
    assert model_data.validator.getLLH(aggLogistic, model_data.train) == approx(0.060, rel=0.1)
    assert model_data.validator.getLLH(aggLogistic, model_data.valid) == approx(0.060, rel=0.1)


def test_agg_mrf_model(model_data):
    regulL2 = 16
    nbSamples = 10000
    nbIter = 50
    memMrf = AggMRFModel(
        model_data.aggdata,
        model_data.features,
        exactComputation=False,  ## Using Gibbs Sampling.  actualy exact=True is broken in latest code
        clicksCfs="*&*",  ## crossfeatures used by P(Y|X) part of the model
        displaysCfs="*&*",  ## crossfeatures used by P(X) part of the model. Here, all pairs + all single .
        nbSamples=nbSamples,  ## Nb Gibbs samples to estimate gradient
        regulL2=1.0,  ## parmeter "lambda_2"
        regulL2Click=regulL2,  ## parmeter "lambda_1"
        sampleFromPY0=True,
        maxNbRowsperGibbsUpdate=50,
    )
    memMrf.fit(nbIter)
    assert model_data.validator.getLLH(memMrf, model_data.train) == approx(0.057, rel=0.1)
    assert model_data.validator.getLLH(memMrf, model_data.valid) == approx(0.057, rel=0.1)
