import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models
)

from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow
import dagshub

dagshub.init(
    repo_owner="subhanseey",
    repo_name="Network-Security",
    mlflow=True
)


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classificationmetric):

        with mlflow.start_run():

            mlflow.log_metric("f1_score", classificationmetric.f1_score)
            mlflow.log_metric("precision", classificationmetric.precision_score)
            mlflow.log_metric("recall_score", classificationmetric.recall_score)

            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model"
            )

    def train_model(self, X_train, y_train, x_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"]
            },

            "Random Forest": {
                "n_estimators": [8, 32, 128, 256]
            },

            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256]
            },

            "Logistic Regression": {},

            "AdaBoost": {
                "learning_rate": [0.1, 0.01, 0.001],
                "n_estimators": [8, 16, 32, 64, 128]
            }
        }

        # ✅ FIX: unpack correctly
        model_report, best_models = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=x_test,
            y_test=y_test,
            models=models,
            param=params
        )

        best_model_score = max(model_report.values())
        best_model_name = max(model_report, key=model_report.get)

        # ✅ FIX: use trained model
        best_model = best_models[best_model_name]

        # ---------- TRAIN METRICS ----------
        y_train_pred = best_model.predict(X_train)

        classification_train_metric = get_classification_score(
            y_true=y_train,
            y_pred=y_train_pred
        )

        self.track_mlflow(best_model, classification_train_metric)

        # ---------- TEST METRICS ----------
        y_test_pred = best_model.predict(x_test)

        classification_test_metric = get_classification_score(
            y_true=y_test,
            y_pred=y_test_pred
        )

        self.track_mlflow(best_model, classification_test_metric)

        # ---------- PREPROCESSOR ----------
        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        # ---------- MODEL WRAPPER ----------
        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=best_model
        )

        os.makedirs("final_model", exist_ok=True)

        # ✅ FIX: save only once
        save_object(
            file_path="final_model/model.pkl",
            obj=network_model
        )

        # ---------- ARTIFACT ----------
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )

        logging.info(f"Model trainer artifact: {model_trainer_artifact}")

        return model_trainer_artifact

    def initiate_model_trainer(self):

        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            return self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        