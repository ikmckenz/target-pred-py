# pylint: disable=invalid-name
"""Script to train the structure to MOA model"""
import argparse
import sys

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.features.build_features import Features
from src.models.models import StructureToMOARFModel, StructureToMOANNModel

parser = argparse.ArgumentParser()

# Parse model type
model_group = parser.add_mutually_exclusive_group(required=True)
model_group.add_argument("--nn", action="store_true", help="Use neural net model")
model_group.add_argument("--rf", action="store_true", help="Use random forrest model")

# Add options for RF model
parser.add_argument("--trees", help="n_estimators for the model", type=int, default=10)

# Add options for NN model
parser.add_argument("--no-gpu", "--no_gpu", action="store_true", help="Train nerual net on cpu", default=False)

args = parser.parse_args()

# Training data
print("Loading training features")
features = Features()
X, y, y_transform = features.load_training_features()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
n_classes = len(y_transform)
n_features = X_train.shape[1]

# Fitting model
print("Fitting model")
if args.rf:
    model = StructureToMOARFModel(y_transform=y_transform, n_estimators=args.trees)
elif args.nn:
    if args.no_gpu:
        model = StructureToMOANNModel(y_transform=y_transform, device="cpu", n_classes=n_classes, n_features=n_features)
    else:
        model = StructureToMOANNModel(y_transform=y_transform, n_classes=n_classes, n_features=n_features)
else:
    raise SyntaxError("Must specify a model type.")

model.train(X_train, y_train)

# Predicting test set
print("Predicting test set")
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

correct = 0
incorrect = 0
top_five, _ = model.predict_top(X_test)
for idx, pred in enumerate(top_five):
    if y_test[idx] in pred:
        correct += 1
    else:
        incorrect += 1

print("Top 5 Accuracy:")
print("{} correct in top 5, {} not in top 5. {:.2%} top 5 accuracy.".format(
    correct, incorrect, (correct/(correct+incorrect))))

# Save model
model.save_model(overwrite=True)
