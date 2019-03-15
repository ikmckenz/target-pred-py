import os
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn import preprocessing


class Features(object):
    """This class turns the processed data into features and results for
    training and also turns SMILES into features for inference"""

    def __init__(self,
                 project_base="../../",
                 processed_data_path="data/processed/",
                 features_filename="features.pickle"):
        self.project_base = project_base
        self.processed_data_path = processed_data_path
        self.features_filename = features_filename

    def build_training_features(self):
        print("Building training features")
        file_loc = self.project_base + self.processed_data_path + "smiles_to_receptor.csv"
        data_set = pd.read_csv(file_loc)
        X = data_set["canonical_smiles"].apply(lambda x:
                                               self.get_numpy_fingerprint_from_smiles(x))
        X = np.stack(X.values)

        le = preprocessing.LabelEncoder().fit(data_set["pref_name"])
        y = le.transform(data_set["pref_name"])
        y_transform = le.classes_

        return X, y, y_transform

    def save_training_features(self, overwrite=False):
        file_loc = self.project_base + self.processed_data_path + self.features_filename
        if not os.path.isfile(file_loc):
            X, y, y_transform = self.build_training_features()
            with open(file_loc, 'wb') as f:
                pickle.dump([X, y, y_transform], f, protocol=4)
            return X, y, y_transform
        elif overwrite:
            X, y, y_transform = self.build_training_features()
            os.remove(file_loc)
            with open(file_loc, 'wb') as f:
                pickle.dump([X, y, y_transform], f, protocol=4)
            return X, y, y_transform
        else:
            print("Not overwriting existing features")

    def load_training_features(self):
        file_loc = self.project_base + self.processed_data_path + self.features_filename
        if os.path.isfile(file_loc):
            with open(file_loc, "rb") as f:
                data = pickle.load(f)
                X = data[0]
                y = data[1]
                y_transform = data[2]
                return X, y, y_transform
        else:
            return self.save_training_features()

    @staticmethod
    def get_numpy_fingerprint_from_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 3)
        finger_container = np.empty(fingerprint.GetNumBits())
        DataStructs.ConvertToNumpyArray(fingerprint, finger_container)
        return finger_container

    @staticmethod
    def list_to_input(list_input):
        arr = np.array(list_input)
        return arr.reshape(1, -1)


if __name__ == "__main__":
    Features().save_training_features(overwrite=True)
