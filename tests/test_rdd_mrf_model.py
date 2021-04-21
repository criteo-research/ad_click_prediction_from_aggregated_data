from pytest import approx
from aggregated_models.agg_mrf_model import AggMRFModel
import pyspark.sql.functions as F


def test_rdd_agg_mrf_model(spark_session, model_data):
    regulL2 = 16
    nbSamples = 10000
    nbIter = 50
    rddMrf = AggMRFModel(
        model_data.aggdata,
        model_data.features,
        exactComputation=False,  # Using Gibbs Sampling.  actualy exact=True is broken in latest code
        clicksCfs="*&*",  # crossfeatures used by P(Y|X) part of the model
        displaysCfs="*&*",  # crossfeatures used by P(X) part of the model. Here, all pairs + all single .
        nbSamples=nbSamples,  # Nb Gibbs samples to estimate gradient
        regulL2=1.0,  # parmeter "lambda_2"
        regulL2Click=regulL2,  # parmeter "lambda_1"
        sampleFromPY0=True,
        maxNbRowsPerSlice=1000,
        sparkSession=spark_session,
    )
    rddMrf.fit(nbIter, 0.05)
    assert model_data.validator.getLLH(rddMrf, model_data.train) == approx(0.057, rel=0.1)
    assert model_data.validator.getLLH(rddMrf, model_data.valid) == approx(0.057, rel=0.1)


def test_rdd_agg_mrf_model_spark_predict(spark_session, spark_model_data):
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
        maxNbRowsPerSlice=1000,
        sparkSession=spark_session,
    )
    rddMrf.fit(nbIter, 0.05)
    predict_df = rddMrf.predictDF(spark_model_data.spark_train, "prediction")
    agg_df = predict_df.agg(F.sum(spark_model_data.label), F.sum("prediction")).collect()
    assert agg_df[0][0] == approx(41637, rel=0.0, abs=0.0)
    assert agg_df[0][1] == approx(42323, rel=0.01, abs=0.01)
    predict_df = rddMrf.predictDF(spark_model_data.spark_valid, "prediction")
    agg_df = predict_df.agg(F.sum(spark_model_data.label), F.sum("prediction")).collect()
    assert agg_df[0][0] == approx(18001, rel=0.0, abs=0.0)
    assert agg_df[0][1] == approx(18288, rel=0.01, abs=0.01)


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
        maxNbRowsPerSlice=1000,
        sparkSession=spark_session,
    )
    rddMrf.fit(nbIter, 0.05)
    assert spark_model_data.spark_validator.getLLH(rddMrf, spark_model_data.spark_train) == approx(0.057, rel=0.1)
    assert spark_model_data.spark_validator.getLLH(rddMrf, spark_model_data.spark_valid) == approx(0.057, rel=0.1)
