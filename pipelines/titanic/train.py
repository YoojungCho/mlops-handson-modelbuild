"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import pandas as pd
import pickle
import joblib

from sklearn import svm #support vector Machine

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Reading train data.")
    train_path = "/opt/ml/input/data/train/train.csv"
    
    path = "/opt/ml/input/data/train/"
    file_list = os.listdir(path)
    print ("train file_list: {}".format(file_list))
    
    df = pd.read_csv(train_path)
    train_X=df[df.columns[1:]]
    train_Y=df[df.columns[:1]]
    
    model=svm.SVC(kernel='rbf',C=1,gamma=0.1)
    model.fit(train_X,train_Y)
    
    joblib.dump(model, '/opt/ml/model/model.h')
    