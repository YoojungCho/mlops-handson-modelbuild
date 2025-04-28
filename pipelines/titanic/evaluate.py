"""Evaluation script for measuring mean squared error."""
import json
import logging
import pathlib
import pickle
import tarfile

import boto3
import pandas as pd
import pickle
import joblib

from sklearn import svm #support vector Machine
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.debug("Loading model.")
    model = joblib.load('./model.h')

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path)
    
    test_X=df[df.columns[1:]]
    test_Y=df[df.columns[:1]]
    
    result = model.predict(test_X)
    
    accuracy = metrics.accuracy_score(result,test_Y)
    print('Accuracy for rbf SVM is ', accuracy)

    # std = np.std(test_X - result)
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": accuracy,
                "standard_deviation": "NaN"
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
