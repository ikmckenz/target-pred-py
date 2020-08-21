"""Contains the classes for dealing with features for the models"""

import os
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn import preprocessing
from multiprocessing import cpu_count, Pool

CORES = cpu_count()

class Features:
    """This class turns the processed data into features and results for
    training and also turns SMILES into features for inference"""

    def __init__(self,
                 project_base="../../",
                 processed_data_path="data/processed/",
                 features_filename="features.pickle"):
        self.project_base = project_base
        self.processed_data_path = processed_data_path
        self.features_filename = features_filename

    def build_training_features(self, parallel=None):
        """Take the smiles to receptor data and create feature vectors"""

        print("Building training features")
        file_loc = self.project_base + self.processed_data_path + "smiles_to_receptor.csv"
        data_set = pd.read_csv(file_loc)
        smiles = data_set["canonical_smiles"]
        if parallel is not None:
            split_smiles = np.array_split(smiles, parallel)
            pool = Pool(CORES)
            X = pd.concat(pool.map(self.get_numpy_fingerprint_from_smiles_series, split_smiles))
            pool.close()
            pool.join()
        else:
            X = smiles.map(self.get_numpy_fingerprint_from_smiles)
        X = np.stack(X.values)

        label_encoder = preprocessing.LabelEncoder().fit(data_set["pref_name"])
        y = label_encoder.transform(data_set["pref_name"])
        y_transform = label_encoder.classes_

        return X, y, y_transform

    def save_training_features(self, overwrite=False, parallel=None):
        """Save the generated features"""

        file_loc = self.project_base + self.processed_data_path + self.features_filename
        if not os.path.isfile(file_loc):  # pylint: disable=no-else-return
            X, y, y_transform = self.build_training_features(parallel=parallel)
            with open(file_loc, 'wb') as f:
                pickle.dump([X, y, y_transform], f, protocol=4)
            return X, y, y_transform
        elif overwrite:
            X, y, y_transform = self.build_training_features(parallel=parallel)
            os.remove(file_loc)
            with open(file_loc, 'wb') as f:
                pickle.dump([X, y, y_transform], f, protocol=4)
            return X, y, y_transform
        else:
            raise IOError("Not overwriting existing features")

    def load_training_features(self):
        """Load saved features"""

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
        """Get Morgan Fingerprint as NumPy vector from SMILES string"""

        mol = Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 3)
        finger_container = np.empty(fingerprint.GetNumBits())
        DataStructs.ConvertToNumpyArray(fingerprint, finger_container)
        return finger_container

    def get_numpy_fingerprint_from_smiles_series(self, smiles_series):
        """Apply get_numpy_fingerprint_from_smiles to a Pandas Series"""

        return smiles_series.map(self.get_numpy_fingerprint_from_smiles)

    @staticmethod
    def list_to_input(list_input):
        """Reshape a list to model input"""

        arr = np.array(list_input)
        return arr.reshape(1, -1)


if __name__ == "__main__":
    Features().save_training_features(overwrite=True, parallel=CORES)
