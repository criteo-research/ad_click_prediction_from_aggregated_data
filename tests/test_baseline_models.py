from pytest import approx, fixture

# baselines
from aggregated_models.basicmodels import LogisticModel, LogisticModelWithCF
from aggregated_models.aggLogistic import AggLogistic


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
