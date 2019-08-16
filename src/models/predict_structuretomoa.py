# pylint: disable=invalid-name
"""Script to predict target for a small molecule"""

import argparse
from src.features.build_features import Features
from src.models.models import StructureToMOAModel

parser = argparse.ArgumentParser()
parser.add_argument("--smiles", help="SMILES code of structure you want to predict", type=str)
args = parser.parse_args()

features = Features()
model = StructureToMOAModel()
model.load_model()

smiles = args.smiles
smiles_features = features.get_numpy_fingerprint_from_smiles(smiles)
model_input = features.list_to_input(smiles_features)
top_output = model.predict_top_pretty(model_input)

print("{} predicted to act on:".format(smiles))
print(top_output)
