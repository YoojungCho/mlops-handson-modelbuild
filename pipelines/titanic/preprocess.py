"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split #training and testing data split

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/train.csv"
    
    print('bucket : ', bucket)
    print('key : ', key)
    print('fn : ', fn)
    
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(fn)
    os.unlink(fn)

    df['Initial']=0
    for i in df:
        df['Initial']=df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
    df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                          ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)

    ## Assigning the NaN Values with the Ceil values of the mean ages
    df.loc[(df.Age.isnull())&(df.Initial=='Mr'),'Age']=33
    df.loc[(df.Age.isnull())&(df.Initial=='Mrs'),'Age']=36
    df.loc[(df.Age.isnull())&(df.Initial=='Master'),'Age']=5
    df.loc[(df.Age.isnull())&(df.Initial=='Miss'),'Age']=22
    df.loc[(df.Age.isnull())&(df.Initial=='Other'),'Age']=46

    df['Embarked'].fillna('S',inplace=True)

    df['Age_band']=0
    df.loc[df['Age']<=16,'Age_band']=0
    df.loc[(df['Age']>16)&(df['Age']<=32),'Age_band']=1
    df.loc[(df['Age']>32)&(df['Age']<=48),'Age_band']=2
    df.loc[(df['Age']>48)&(df['Age']<=64),'Age_band']=3
    df.loc[df['Age']>64,'Age_band']=4

    df['Family_Size']=0
    df['Family_Size']=df['Parch']+df['SibSp']#family size
    df['Alone']=0
    df.loc[df.Family_Size==0,'Alone']=1#Alone

    df['Fare_Range']=pd.qcut(df['Fare'],4)

    df['Fare_cat']=0
    df.loc[df['Fare']<=7.91,'Fare_cat']=0
    df.loc[(df['Fare']>7.91)&(df['Fare']<=14.454),'Fare_cat']=1
    df.loc[(df['Fare']>14.454)&(df['Fare']<=31),'Fare_cat']=2
    df.loc[(df['Fare']>31)&(df['Fare']<=513),'Fare_cat']=3

    df['Sex'].replace(['male','female'],[0,1],inplace=True)
    df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
    df['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

    df.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)
    
    logger.info("Splitting %d rows of data into train, test datasets.", len(df))
    train,test=train_test_split(df,test_size=0.3,random_state=0,stratify=df['Survived'])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", index=False)
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", index=False)