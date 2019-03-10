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
                 processed_data_path="data/processed/smiles_to_receptor.csv"):
        self.project_base = project_base
        self.processed_data_path = processed_data_path

    def build_training_features(self):
        data_set = pd.read_csv(self.project_base + self.processed_data_path)
        data_set = data_set[:2]
        X = data_set["canonical_smiles"].apply(lambda x:
                                               self.get_numpy_fingerprint_from_smiles(x))

        le = preprocessing.LabelEncoder().fit(data_set["pref_name"])
        y = le.transform(data_set["pref_name"])
        y_transform = le.classes_

        return X, y, y_transform

    @staticmethod
    def get_numpy_fingerprint_from_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 3)
        finger_container = np.empty(fingerprint.GetNumBits())
        DataStructs.ConvertToNumpyArray(fingerprint, finger_container)
        return finger_container
