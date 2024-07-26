import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import os
from src.utils import save_file_as_pickle

@dataclass
class DataTransformationConfig:
    dt_pickle_path=os.path.join("artifacts/pickle","data_transformation.pkl")

class DataTransformation:
    def __init__(self):
        self.dt_pickle=DataTransformationConfig()

    def data_transform(self):
        numerical_columns = ["Amount"]
        numerical_column_pipeline=Pipeline(steps=[
            ("standardscaler",StandardScaler())
        ])
        preprocessing=ColumnTransformer([
            ("numerical_columns_pipeline",numerical_column_pipeline,numerical_columns)
        ],remainder="passthrough")

        return preprocessing

    def initiate_data_transformation(self,train_dataset_path,test_dataset_path):
        try:
            train_dataset=pd.read_csv(train_dataset_path)
            test_dataset=pd.read_csv(test_dataset_path)

            train_dataset.rename(columns={"Class":"fraudulent"},inplace=True)
            test_dataset.rename(columns={"Class":"fraudulent"},inplace=True)



            columns_to_drop=["index","fraudulent"]
            target_feature="fraudulent"

            xtrain=train_dataset.drop(columns=columns_to_drop)
            ytrain=train_dataset[target_feature]

            xtest=test_dataset.drop(columns=columns_to_drop)
            ytest=test_dataset[target_feature]


            transform_data=self.data_transform()
            save_file_as_pickle(self.dt_pickle.dt_pickle_path,transform_data)


            transform_xtrain=transform_data.fit_transform(xtrain)
            transform_xtest=transform_data.transform(xtest)

            transformed_train_dataset=np.c_[transform_xtrain,np.array(ytrain)]
            transformed_test_dataset=np.c_[transform_xtest,np.array(ytest)]

            return transformed_train_dataset,transformed_test_dataset

        except Exception as e:
            raise e






