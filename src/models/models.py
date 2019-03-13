import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class StructureToMOAModel(object):
    def __init__(self,
                 y_transform=None,
                 project_base="../../",
                 model_save_path="models/structuretomoa_model.pickle"):

        self.project_base = project_base
        self.model_save_path = model_save_path

        self.model = RandomForestClassifier()
        self.y_transform = y_transform
        self.trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, data):
        return self.model.predict(data)

    def pred_to_label(self, pred):
        if np.isscalar(pred):
            return self.y_transform[pred]
        else:
            return [self.y_transform[x] for x in pred]

    def save_model(self, overwrite=False):
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
        file_loc = self.project_base + self.model_save_path
        if os.path.isfile(file_loc):
            with open(file_loc, "rb") as f:
                data = pickle.load(f)
                self.model = data[0]
                self.y_transform = data[1]
        else:
            print("Could not find model to load")

    def _save_model(self, file_loc):
        with open(file_loc, 'wb') as f:
            pickle.dump([self.model, self.y_transform], f)
