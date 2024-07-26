import pandas as pd
import os
from src.utils import load_pickle_file
class Prediction:
    def __init__(self):
        pass
    def initiate_prediction(self,feature):
        try:
            data_transformation_pickle_path=os.path.join("artifacts/pickle","data_transformation.pkl")
            model_pickle_path=os.path.join("artifacts/pickle","model.pkl")
            preprocessor=load_pickle_file(data_transformation_pickle_path)
            model=load_pickle_file(model_pickle_path)

            processed_data=preprocessor.transform(feature)
            output=model.predict(processed_data)
            return output
        except Exception as e:
            raise e



class Features:
    def __init__(self):
        pass
    def initiate_features_to_dataframe(self):
        features_to_dictionary={

        }

        feature_to_df=pd.DataFrame(features_to_dictionary)
        return feature_to_df