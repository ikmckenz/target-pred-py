"""The machine learning models"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class StructureToMOARFModel:
    """Defines the random forest classifier model from structure to MOA"""

    def __init__(self,
                 y_transform=None,
                 project_base="../../",
                 model_save_path="models/structuretomoa_model.pickle",
                 n_estimators=10):

        self.project_base = project_base
        self.model_save_path = model_save_path

        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            n_jobs=-1)
        self.y_transform = y_transform
        self.trained = False

    def train(self, X, y):
        """Train the model

        Args:
            X: training input samples
            y: training target values

        """
        self.model.fit(X, y)
        self.trained = True

    def predict(self, data):
        """Predict a MOA from a structure

        Args:
            data: The structure to predict

        """
        return self.model.predict(data)

    def predict_top(self, data, n_outputs=5):
        """Predict top n MOAs from a structure

        Args:
            data: The structure to predict
            n_outputs: The number of predictions

        Returns:
            top: list of numeric category predictions
            probabilities: list of probabilities
        """
        model_output = self.model.predict_proba(data)
        top = np.argsort(model_output)[:, ::-1][:, :n_outputs]
        probabilities = np.take_along_axis(model_output, top, 1)
        return top, probabilities

    def predict_top_pretty(self, data, n_outputs=5):
        """Predict top n MOAs from a single structure

        Args:
            data: The structure to predict
            n_outputs: The number of predictions

        Returns:
            pd.DataFrame: A pretty dataframe of labels and their probabilities
        """
        top, probabilities = self.predict_top(data, n_outputs)
        labels = [self.pred_to_label(x) for x in top[0]]
        return pd.DataFrame({"target": labels, "probability": probabilities[0]})

    def pred_to_label(self, pred):
        """Take a model prediction and return the original label"""
        if np.isscalar(pred):
            return self.y_transform[pred]
        return [self.y_transform[x] for x in pred]

    def save_model(self, overwrite=False):
        """Save the trained model"""
        if self.trained:
            file_loc = self.project_base + self.model_save_path
            if not os.path.isfile(file_loc):
                self._save_model(file_loc)
            elif overwrite:
                os.remove(file_loc)
                self._save_model(file_loc)
            else:
                print("Not overwriting existing model")
        else:
            print("Not saving: Model is untrained")

    def load_model(self):
        """Load a saved model"""
        file_loc = self.project_base + self.model_save_path
        if os.path.isfile(file_loc):
            with open(file_loc, "rb") as f:
                data = pickle.load(f)
                self.model = data[0]
                self.y_transform = data[1]
        else:
            print("Could not find model to load")

    def _save_model(self, file_loc):
        """Private function to pickle the model"""
        with open(file_loc, 'wb') as f:
            pickle.dump([self.model, self.y_transform], f, protocol=4)

