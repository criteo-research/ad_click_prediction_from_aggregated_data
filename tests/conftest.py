import pytest
import pandas as pd
import getpass
from thx.hadoop import spark_config_builder
from aggregated_models.validation import MetricsComputer
from aggregated_models.aggdataset import AggDataset
import uuid


@pytest.fixture(scope="session")
def spark_session():
    ss = spark_config_builder.create_local_spark_session(
        "test-session", sql_num_partitions=1, properties=[("spark.ui.enabled", "false")]
    )
    tmp_fold = str(uuid.uuid4())

    checkpoint_dir = f"/tmp/{getpass.getuser()}/load/{tmp_fold}/"
    ss.sparkContext.setCheckpointDir(checkpoint_dir)

    yield ss
    ss.stop()


class ModelTestData:
    def __init__(self, spark_session=None):
        self.train = pd.read_parquet("./tests/resources/small_train.parquet")
        self.valid = pd.read_parquet("./tests/resources/small_valid.parquet")
        self.features = ["cat1", "cat4", "cat6", "cat8", "cat9"]
        self.label = "click"

        if spark_session is not None:
            self.spark_train = spark_session.createDataFrame(self.train)
        else:
            self.spark_train = None

        self.validator = MetricsComputer(self.label)

        # parameters for of the privacy protecting noise.
        self.epsilon = None  # Set to None to get no noise.
        self.delta = None
        self.aggdata = AggDataset(
            dataframe=(self.train if self.spark_train is None else self.spark_train),
            features=self.features,
            label=self.label,
            epsilon0=self.epsilon,
            delta=self.delta,
            maxNbModalities=10000,
        )


@pytest.fixture(scope="session")
def model_data():
    return ModelTestData()


@pytest.fixture(scope="session")
def spark_model_data(spark_session):
    return ModelTestData(spark_session)
