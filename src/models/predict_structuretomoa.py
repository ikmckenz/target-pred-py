# pylint: disable=invalid-name
"""Script to predict target for a small molecule"""

import argparse
from src.features.build_features import Features
from src.models.models import StructureToMOARFModel, StructureToMOANNModel

parser = argparse.ArgumentParser()
parser.add_argument("--smiles", help="SMILES code of structure you want to predict", type=str)
model_group = parser.add_mutually_exclusive_group(required=True)
model_group.add_argument("--nn", action="store_true", help="Use neural net model")
model_group.add_argument("--rf", action="store_true", help="Use random forrest model")
args = parser.parse_args()

features = Features()
if args.rf:
    model = StructureToMOARFModel()
elif args.nn:
    model = StructureToMOANNModel()
else:
    raise SyntaxError("Must choose model type.")

model.load_model()

smiles = args.smiles
smiles_features = features.get_numpy_fingerprint_from_smiles(smiles)
model_input = features.list_to_input(smiles_features)
top_output = model.predict_top_pretty(model_input)

print("{} predicted to act on:".format(smiles))
print(top_output)
