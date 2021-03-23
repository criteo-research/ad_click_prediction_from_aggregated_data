import pytest
from thx.hadoop import spark_config_builder


@pytest.fixture(scope="session")
def spark_session():
    ss = spark_config_builder.create_local_spark_session(
        "test-session", sql_num_partitions=1, properties=[("spark.ui.enabled", "false")]
    )
    yield ss
    ss.stop()
