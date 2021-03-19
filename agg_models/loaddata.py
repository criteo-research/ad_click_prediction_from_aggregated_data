import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass
import dataclasses


# Loading public "Criteo attribution dataset"
datapath = "../data/"


def download_dataset():
    if not os.path.exists(datapath):
        print("creating ../data")
        os.mkdir(datapath)
    zipfilename = "criteo-research-attribution-dataset.zip"
    print("downloading dataset")
    urllib.request.urlretrieve("http://go.criteo.net/" + zipfilename, datapath + zipfilename)
    print("unzipping")
    with zipfile.ZipFile(datapath + zipfilename, "r") as zip_ref:
        zip_ref.extractall(datapath + "criteo_attribution_dataset")


@dataclass
class DatasetSpec:
    dataset_path: str
    labels: List[str]
    features: List[str]
    nb_rows: int
    fileformat: str


criteo_attribution_dataset = DatasetSpec(
    dataset_path=datapath + "/criteo_attribution_dataset/criteo_attribution_dataset.tsv.gz",
    labels=["click"],  # + ["conversion", "attribution"],
    features=[
        "campaign",
        "time_since_last_click",
        "cat1",
        "cat2",
        "cat3",
        "cat4",
        "cat5",
        "cat6",
        "cat7",
        "cat8",
        "cat9",
    ],
    nb_rows=16468027,
    fileformat="csv",
)


criteo_attribution_dataset_small = dataclasses.replace(
    criteo_attribution_dataset, features=["cat1", "cat4", "cat6", "cat8", "cat9"]
)


criteo_terabyte_dataset = DatasetSpec(
    dataset_path=datapath + "/criteo_terabyte_dataset_small/criteo_tb_sample_1_1000_seed_42_small.gz.parquet",
    labels=["label"],
    features=[
        "integer_feature_1",
        "integer_feature_2",
        "integer_feature_3",
        "integer_feature_4",
        "integer_feature_5",
        "integer_feature_6",
        "integer_feature_7",
        "integer_feature_8",
        "integer_feature_9",
        "integer_feature_10",
        "integer_feature_11",
        "integer_feature_12",
        "integer_feature_13",
        "categorical_feature_1",
        "categorical_feature_2",
        "categorical_feature_3",
        "categorical_feature_4",
        "categorical_feature_5",
        "categorical_feature_6",
        "categorical_feature_7",
        "categorical_feature_8",
        "categorical_feature_9",
        "categorical_feature_10",
        "categorical_feature_11",
        "categorical_feature_12",
        "categorical_feature_13",
        "categorical_feature_14",
        "categorical_feature_15",
        "categorical_feature_16",
        "categorical_feature_17",
        "categorical_feature_18",
        "categorical_feature_19",
        "categorical_feature_20",
        "categorical_feature_21",
        "categorical_feature_22",
        "categorical_feature_23",
        "categorical_feature_24",
        "categorical_feature_25",
        "categorical_feature_26",
    ],
    nb_rows=4370644,
    fileformat="parquet",
)

criteo_terabyte_dataset_small = dataclasses.replace(
    criteo_terabyte_dataset,
    features=[
        "integer_feature_10",
        "categorical_feature_6",
        "categorical_feature_13",
        "categorical_feature_17",
        "categorical_feature_19",
        "categorical_feature_26",
    ],
)


def run(
    dataset: DatasetSpec,
    samplingRate=0.01,
    validationRate=0.3,
    splitOnDate=False,
    seed=0,
    verbose=False,
):
    dataset_path = dataset.dataset_path
    if not os.path.exists(dataset_path):
        download_dataset()

    if seed is not None:
        np.random.seed(seed)  # to get the same dataset and consistent results

    usecols = dataset.labels + dataset.features + (["day"] if splitOnDate else list())
    n = dataset.nb_rows  # number of lines in the whole file
    skip = np.random.rand(n) < samplingRate
    if dataset.fileformat == "csv":
        df = pd.read_csv(dataset_path, sep="\t", compression="gzip", usecols=usecols)
    else:
        df = pd.read_parquet(dataset_path, columns=usecols)
    df = df[skip]
    for c in df.columns:
        default = df[c].min() - 1 if df[c].dtype in ["int32", "int64", "float64"] else ""
        df[c] = df[c].fillna(default)
    df["c"] = 1

    def bucketize(x, logbase=2.0, offset=0.0):
        if not x:
            return -200
        x += offset
        if x <= 0:
            return -100
        return int(np.log(x) / np.log(2))

    if "time_since_last_click" in df.columns:
        df.time_since_last_click = df.time_since_last_click.apply(lambda x: bucketize(x, 2.0, 3600 * 6))
    # Set constant column for row count
    df["c"] = 1
    # Split Train / Validation
    n = len(df)
    if splitOnDate:
        istrain = df["day"] < splitOnDate
    else:
        istrain = np.random.rand(n) > validationRate
    valid = df[~istrain]
    train = df[istrain]
    if verbose:
        for x in dataset.labels:
            print(x, sum(df[x]) / len(df))
        for x in dataset.features:
            print(x, len(df[[x, "c"]].groupby(x).agg(sum)))
    return train, valid, dataset.features


def getDataset(name, forceSamplingRate=None, splitOnDate=None):
    """Load one of the 3 dataset described in the paper:
    - "small" is sampled at 1%, and use only 5 smallest features, to get fast running experiments.
    - "sampled" contains 1% of examples with all (11) features.
       Recommended at least 16Go to run expriments
    - "full" contains all examples (15M) and 11 features.
       some unfrequent modalities (unfrequent in train) are grouped together to limit a bit the memory footprint.
       Training models is *slow*, experiments requires 32Go RAM.
    """
    if name not in ["small", "sampled", "full", "small_tb", "medium_tb", "full_tb"]:
        raise Exception(f"{name} is not a known dataset")

    if "full" in name:
        samplingRate = 1.0
    elif "medium" in name:
        samplingRate = 0.1
    else:
        samplingRate = 0.01
    samplingRate = forceSamplingRate or samplingRate

    if "_tb" in name:
        dataset = criteo_terabyte_dataset_small
    elif name == "small":
        # features with smallest modalities counts
        dataset = criteo_attribution_dataset_small
    else:
        dataset = criteo_attribution_dataset

    print(f"Sampling ratio :{samplingRate}")
    train, valid, allvars = train, valid, allvars = run(dataset, samplingRate, splitOnDate=splitOnDate, verbose=False)
    if name == "full":
        # on dataset-full, some features have a marge number of modalities with very few events.
        # there is not much to learn from those modalities,
        # so they were grouped together in a single "other" modality to limit (a bit) the memory
        # footprint of learning the models.
        # We used the modalities count on the trainset to make sure this grouping does not leak information on validation data.
        minNbModalities = 20
        for f in ["cat7", "cat3"]:
            b = train[["c", f]].groupby(f).sum().reset_index()
            keptcat7 = set(b[b["c"] > minNbModalities][f].values)

            train[f] = train[f].apply(lambda x: x if x in keptcat7 else 0)
            valid[f] = valid[f].apply(lambda x: x if x in keptcat7 else 0)
    features = dataset.features
    print(f"Nb train samples: {len(train)} , Nb valid samples: {len(valid)}  ")
    print(f"features:{features}")

    label = dataset.labels[0]
    cols = features + [label] + (["day"] if splitOnDate else list())
    train = train[cols]
    valid = valid[cols]

    return train, valid, features, label
