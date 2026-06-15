import os
import sys
import yaml
import numpy as np
import dill
import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from networksecurity.exception.exception import NetworkSecurityException


# =========================
# YAML UTILITIES
# =========================

def read_yaml_file(file_path: str):
    try:
        with open(file_path, "rb") as file:
            return yaml.safe_load(file)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        mode = "w" if replace else "a"

        with open(file_path, mode) as file:
            yaml.dump(content, file)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# =========================
# OBJECT SERIALIZATION
# =========================

def save_object(file_path: str, obj: object):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str):
    try:
        with open(file_path, "rb") as file:
            return dill.load(file)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# =========================
# NUMPY UTILITIES
# =========================

def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Save numpy array to file
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            np.save(file, array)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str):
    """
    Load numpy array from file
    """
    try:
        return np.load(file_path)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# =========================
# MODEL EVALUATION
# =========================

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_models = {}

        for model_name, model in models.items():

            params = param.get(model_name, {})

            gs = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=3,
                n_jobs=-1,
                verbose=1
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_models[model_name] = best_model

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = accuracy_score(y_train, y_train_pred)
            test_score = accuracy_score(y_test, y_test_pred)

            logging.info(
                f"{model_name} -> "
                f"Train Accuracy: {train_score:.4f}, "
                f"Test Accuracy: {test_score:.4f}"
            )

            report[model_name] = test_score

        return report, best_models

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    