from pytest import fixture

import numpy as np
from pyspark.sql.session import SparkSession

from aggregated_models.aggdataset import AggDataset
from aggregated_models.agg_mrf_model import AggMRFModel


@fixture(scope="module")
def agg_data(spark_session: SparkSession) -> AggDataset:
    features = ["cat1", "cat2", "cat3"]
    label_col = "label"
    train = spark_session.createDataFrame(
        data=[
            ["hello", "world", "!", 1],
            ["i", "love", "data", 0],
            ["hello", "world", "!", 1],
            ["i", "am", "!", 0],
        ],
        schema=features + [label_col],
    )
    return AggDataset(features=features, dataframe=train, label=label_col, maxNbModalities=None)


def test_agg_data_gives_correct_aggregations(agg_data: AggDataset):
    # Number of queries for N features = N(N+1)/2
    assert len(agg_data.aggDisplays) == 6
    assert len(agg_data.aggClicks) == 6

    # ["hello", "i"]
    np.testing.assert_array_equal(agg_data.aggDisplays["cat1"].Data, np.array([2.0, 2.0, 0.0]))
    # ["am", "love", "world"]
    np.testing.assert_array_equal(agg_data.aggDisplays["cat2"].Data, np.array([1.0, 1.0, 2.0, 0.0]))
    # ["!", "data"]
    np.testing.assert_array_equal(agg_data.aggDisplays["cat3"].Data, np.array([3.0, 1.0, 0.0]))

    # ["hello", "i"]
    np.testing.assert_array_equal(agg_data.aggClicks["cat1"].Data, np.array([2.0, 0.0, 0.0]))
    # ["am", "love", "world"]
    np.testing.assert_array_equal(agg_data.aggClicks["cat2"].Data, np.array([0.0, 0.0, 2.0, 0.0]))
    # ["!", "data"]
    np.testing.assert_array_equal(agg_data.aggClicks["cat3"].Data, np.array([2.0, 0.0, 0.0]))

    # ["hello", "i"] x ["am", "love", "world"]
    # ["hello" x "world"] + ["i" x "love"] + ["i" x "am"] are present
    #          2                 1               1           times
    #  idx = 0+3*2 = 6     idx = 1+3*1 = 4  idx = 1+3*0 = 1
    np.testing.assert_array_equal(
        agg_data.aggDisplays["cat1&cat2"].Data, np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )

    # ["hello", "i"] x ["!", "data"]
    # ["hello" x "!"] + ["i" x "data"] + ["i" x "!"] are present
    #          2             1                1        times
    #  idx = 0+3*0 = 0   idx = 1+3*1 = 4  idx = 1+3*0 = 1
    np.testing.assert_array_equal(
        agg_data.aggDisplays["cat1&cat3"].Data, np.array([2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    )

    # ["am", "love", "world"] x ["!", "data"]
    # ["world" x "!"] + ["love" x "data"] + ["am" x "!"] are present
    #          2                1                    1        times
    #  idx = 2+4*0 = 2   idx = 1+4*1 = 5  idx = 0+4*0 = 0
    np.testing.assert_array_equal(
        agg_data.aggDisplays["cat2&cat3"].Data, np.array([1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )


def test_in_memory_mrf_model_runs_on_fit(agg_data: AggDataset):
    regulL2 = 16
    nbSamples = 10000
    nbIter = 3
    memMrf = AggMRFModel(
        agg_data,
        agg_data.features,
        exactComputation=False,
        clicksCfs="*&*",
        displaysCfs="*&*",
        nbSamples=nbSamples,
        regulL2=1.0,
        regulL2Click=regulL2,
        sampleFromPY0=True,
        maxNbRowsPerSlice=50,
    )
    memMrf.fit(nbIter)


def test_rdd_mrf_model_runs_on_spark_fit(spark_session, agg_data: AggDataset):
    regulL2 = 16
    nbSamples = 10000
    nbIter = 3
    rddMrf = AggMRFModel(
        agg_data,
        agg_data.features,
        exactComputation=False,
        clicksCfs="*&*",
        displaysCfs="*&*",
        nbSamples=nbSamples,
        regulL2=1.0,
        regulL2Click=regulL2,
        sampleFromPY0=True,
        maxNbRowsPerSlice=1000,
        sparkSession=spark_session,
    )
    rddMrf.fit(nbIter)
