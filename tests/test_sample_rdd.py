import pytest
import pandas as pd
import numpy as np
from aggregated_models.aggdataset import AggDataset
from aggregated_models.SampleRdd import SampleRdd


@pytest.fixture(scope="module")
def aggdata() -> AggDataset:
    features = ["int1", "int2", "int3"]
    label_col = "label"
    train = pd.DataFrame(
        data=[
            [0, 0, 0, 10],
            [1, 0, 1, 10],
            [2, 2, 1, 100],
            [3, 1, 0, 10],
        ],
        columns=features + [label_col],
    )
    return AggDataset(train, features=features, label=label_col, maxNbModalities=None)


def test_sample_rdd_construction(spark_session, aggdata):
    np.random.seed(42)
    nbSamples = 1000
    maxNbRowsPerSlice = 10
    sampleFromPY0 = False
    decollapseGibbs = False
    features = aggdata.features

    projections = {feature: aggdata.aggDisplays[feature] for feature in features}

    sample = SampleRdd(
        [projections[feature] for feature in features],
        spark_session,
        None,
        None,
        nbSamples,
        decollapseGibbs,
        sampleFromPY0,
        maxNbRowsPerSlice,
    )

    rows = sample.get_rows()
    print(rows)
    assert rows.shape == (1000, 3)
    assert rows.min() == 0
    assert rows[:, 0].max() == 3
    assert rows[:, 1].max() == 2
    assert rows[:, 2].max() == 1
