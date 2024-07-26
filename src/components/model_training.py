import os.path
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dataclasses import dataclass
from src.utils import save_file_as_pickle


@dataclass
class ModelTrainingConfig:
    model_pickle = os.path.join("artifacts/pickle", "model.pkl")


class ModelTraining:
    def __init__(self):
        self.model_config = ModelTrainingConfig()

    def initiate_model_training(self, transformed_train_dataset, transformed_test_dataset):
        try:
            xtrain = transformed_train_dataset[:, :-1]
            ytrain = transformed_train_dataset[:, -1]

            xtest = transformed_test_dataset[:, :-1]
            true_value = transformed_test_dataset[:, -1]

            models = {
                "logistics_regression": LogisticRegression(),
                "Svm": SVC(),
                "Decision_tree": DecisionTreeClassifier(),
                "Random_forest": RandomForestClassifier(),
                "Adaboost": AdaBoostClassifier(),
                "Gradient_boost": GradientBoostingClassifier()
            }
            model_report = {}
            for model_name, model in models.items():
                model.fit(xtrain, ytrain)
                predicted_value = model.predict(xtest)
                model_accuracy = accuracy_score(true_value, predicted_value)
                #model_recall = recall_score(true_value, predicted_value)
                #model_precision = precision_score(true_value, predicted_value)
                #print("model name",model_name,"has accuracy of:",model_accuracy,"with precision score of:",model_precision," and recall score of: ",model_recall)
                model_report[model_name] = model_accuracy

            best_model_name = max(model_report, key=model_report.get)
            best_model_accuracy = max(model_report.values())

            best_model=models[best_model_name]

            save_file_as_pickle(self.model_config.model_pickle,best_model)

        except Exception as e:
            raise e
#%%
