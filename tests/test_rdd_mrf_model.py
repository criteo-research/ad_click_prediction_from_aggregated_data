from pytest import approx
from aggregated_models.agg_mrf_model import AggMRFModel


def test_spark_agg_mrf_model(spark_session, model_data):
    regulL2 = 16
    nbSamples = 10000
    nbIter = 50
    memMrf = AggMRFModel(
        model_data.aggdata,
        model_data.features,
        exactComputation=False,  # Using Gibbs Sampling.  actualy exact=True is broken in latest code
        clicksCfs="*&*",  # crossfeatures used by P(Y|X) part of the model
        displaysCfs="*&*",  # crossfeatures used by P(X) part of the model. Here, all pairs + all single .
        nbSamples=nbSamples,  # Nb Gibbs samples to estimate gradient
        regulL2=1.0,  # parmeter "lambda_2"
        regulL2Click=regulL2,  # parmeter "lambda_1"
        sampleFromPY0=True,
        maxNbRowsperGibbsUpdate=100,
        sparkSession=spark_session,
    )
    memMrf.fit(nbIter, 0.05)
    assert model_data.validator.getLLH(memMrf, model_data.train) == approx(0.057, rel=0.1)
    assert model_data.validator.getLLH(memMrf, model_data.valid) == approx(0.057, rel=0.1)


def test_spark_agg_mrf_model_spark_validation(spark_session, spark_model_data):
    regulL2 = 16
    nbSamples = 10000
    nbIter = 50
    rddMrf = AggMRFModel(
        spark_model_data.aggdata,
        spark_model_data.features,
        exactComputation=False,  # Using Gibbs Sampling.  actualy exact=True is broken in latest code
        clicksCfs="*&*",  # crossfeatures used by P(Y|X) part of the model
        displaysCfs="*&*",  # crossfeatures used by P(X) part of the model. Here, all pairs + all single .
        nbSamples=nbSamples,  # Nb Gibbs samples to estimate gradient
        regulL2=1.0,  # parmeter "lambda_2"
        regulL2Click=regulL2,  # parmeter "lambda_1"
        sampleFromPY0=True,
        maxNbRowsperGibbsUpdate=100,
        sparkSession=spark_session,
    )
    rddMrf.fit(nbIter, 0.05)
    assert spark_model_data.spark_validator.getLLH(rddMrf, spark_model_data.spark_train) == approx(0.057, rel=0.1)
    assert spark_model_data.spark_validator.getLLH(rddMrf, spark_model_data.spark_valid) == approx(0.057, rel=0.1)
