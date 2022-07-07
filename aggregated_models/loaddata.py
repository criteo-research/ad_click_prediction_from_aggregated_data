import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from typing import List
from dataclasses import dataclass
import dataclasses
from sklearn.model_selection import train_test_split
from itertools import combinations 
import matplotlib.pyplot as plt


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

    train, valid = get_dataset(name, dataset, samplingRate, splitOnDate)

    features = dataset.features

    label = dataset.labels[0]
    cols = features + [label] + (["day"] if splitOnDate else list())
    train = train[cols]
    valid = valid[cols]

    print(f"Sampling ratio :{samplingRate}")
    print(f"Nb train samples: {len(train)} , Nb valid samples: {len(valid)}  ")
    print(f"features:{features}")

    return train, valid, features, label


def get_dataset(name, dataset, samplingRate, splitOnDate):
    train_path = f"{name}_train_{samplingRate}_{splitOnDate}.parquet"
    valid_path = f"{name}_valid_{samplingRate}_{splitOnDate}.parquet"
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        train, valid = build_dataset(name, dataset, samplingRate, splitOnDate)
        train.to_parquet(train_path)
        valid.to_parquet(valid_path)
    else:
        train = pd.read_parquet(train_path)
        valid = pd.read_parquet(valid_path)
    return train, valid


def build_dataset(name, dataset, samplingRate, splitOnDate):
    train, valid, allvars = run(dataset, samplingRate, splitOnDate=splitOnDate, verbose=False)
    if name == "full":
        # on dataset-full, some features have a marge number of modalities with very few events.
        # there is not much to learn from those modalities,
        # so they were grouped together in a single "other" modality to limit (a bit) the memory
        # footprint of learning the models.
        # We used the modalities count on the trainset to make sure
        # this grouping does not leak information on validation data.
        minNbModalities = 20
        for f in ["cat7", "cat3"]:
            b = train[["c", f]].groupby(f).sum().reset_index()
            keptcat7 = set(b[b["c"] > minNbModalities][f].values)

            train[f] = train[f].apply(lambda x: x if x in keptcat7 else 0)
            valid[f] = valid[f].apply(lambda x: x if x in keptcat7 else 0)
    return train, valid




##  Code for loading "adult" and "baking" datasets.


def quant_centroid(array):
    unic=np.unique(array)
    decimal=(max(unic)-min(unic))/10
    centroid=[round(min(unic) + i*decimal) for i in range(1,11)]
    return(centroid)

def from_num_to_cat(array,centroid):
    new_array=np.zeros_like(array)
    for i in range(len(array)):
        if array[i]<=centroid[0]:
            new_array[i]=centroid[0]
        if array[i]>=centroid[8]:
            new_array[i]=centroid[-1]
        for j in range(8):
            if array[i]>=centroid[j] and array[i]<=centroid[j+1] :
                new_array[i]=centroid[j+1]
    return(new_array) 


def download_banking_dataset():
    import os.path
    if os.path.isfile('data/bank_dataset/bank-additional/bank-additional-full.csv'):
        return
    print("downloading banking dataset")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    import urllib.request
    urllib.request.urlretrieve(url, "data/bank_dataset.zip")
    print("uzipping banking dataset")
    import zipfile
    with zipfile.ZipFile("data/bank_dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("data/bank_dataset")
    
    
def load_banking_dataset():
    download_banking_dataset()
    Bdata = pd.read_csv('data/bank_dataset/bank-additional/bank-additional-full.csv', sep=';', na_values='?',engine='python')
    Bdata.rename(columns={'y': 'label'}, inplace=True)
    Bdata["label"]  = 1 * (Bdata["label"] == "yes" )
    Bfeatures=list(Bdata.columns[:-1])
    if "label" in Bfeatures:
        raise("label found in features list")
    Btrain, Btest = train_test_split(Bdata, test_size=0.2)
    Btrain=Btrain.copy()
    Btest=Btest.copy()

    # Quantifying numerical data
    Bnum_attributes = Btrain.select_dtypes(include=['float','int'])
    for num in Bnum_attributes.columns[:-1]:
        centroid=quant_centroid(Btrain[num])
        Btrain[num]=from_num_to_cat(Btrain[num].values,centroid)
        Btest[num]=from_num_to_cat(Btest[num].values,centroid)    
                             
    # Hashing data
    dfBtrain=Btrain.copy()
    dfBtest=Btest.copy()

    for feat in Bfeatures:
        dfBtrain[feat]=dfBtrain[feat].apply(hash)
        dfBtest[feat]=dfBtest[feat].apply(hash)
        
    return  dfBtrain, dfBtest, Bfeatures


def download_adult_dataset():
    import os.path
    import urllib.request
    if not os.path.isfile('adult_data.csv'):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        print("downloading from " + url)
        urllib.request.urlretrieve(url, "adult_data.csv")
    if not os.path.isfile('adult_data_test.csv'):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        print("downloading from " + url)
        urllib.request.urlretrieve(url, "adult_data_test.csv")

    
def load_adult_dataset():
    download_adult_dataset()
    columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship",
              "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "label"]
    Atrain = pd.read_csv('adult_data.csv', names=columns, 
                 sep=' *, *', na_values='?',engine='python')
    Atest  = pd.read_csv('adult_data_test.csv', names=columns, 
                 sep=' *, *',skiprows=1, na_values='?',engine='python')        
    
    Atrain["label"] = 1 * (Atrain["label"] .isin([">50K",">50K."]) )
    Atest["label"]  = 1 * (Atest["label"].isin([">50K",">50K."]) )

    # Complete missing Data
    #  cols=['workClass','occupation','native-country']
    #  Atrain[cols]=Atrain[cols].fillna(Atrain.mode().iloc[0])
    #  Atest[cols]=Atest[cols].fillna(Atest.mode().iloc[0])        

    Afeatures=list(Atrain.columns[:-1])
    # Quantifying numerical data
    Anum_attributes = Atrain.select_dtypes(include=['float','int'])
    for num in Anum_attributes.columns[:-1]:
        centroid=quant_centroid(Atrain[num])
        Atrain[num]=from_num_to_cat(Atrain[num].values,centroid)
        Atest[num]=from_num_to_cat(Atest[num].values,centroid)    
    
    # Hashing data
    dfAtrain=Atrain.copy()
    dfAtest=Atest.copy()

    for feat in Afeatures:
        dfAtrain[feat]=dfAtrain[feat].apply(hash)
        dfAtest[feat]=dfAtest[feat].apply(hash)
    
    return dfAtrain, dfAtest, Afeatures
    