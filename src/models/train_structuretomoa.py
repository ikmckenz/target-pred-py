# pylint: disable=invalid-name
"""Script to train the structure to MOA model"""
import argparse

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.features.build_features import Features
from src.models.models import StructureToMOAModel

parser = argparse.ArgumentParser()
parser.add_argument("--trees", help="n_estimators for the model", type=int, default=10)
args = parser.parse_args()

# Training data
print("Loading training features")
features = Features()
X, y, y_transform = features.load_training_features()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Fitting model
print("Fitting model")
model = StructureToMOAModel(y_transform=y_transform, n_estimators=args.trees)
model.train(X_train, y_train)

# Predicting test set
print("Predicting test set")
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

correct = 0
incorrect = 0
for idx, x in enumerate(X_test):
    top_five = model.predict_top(x)
    if y_test[idx] in top_five:
        correct += 1
    else:
        incorrect += 1

print("Top 5 Accuracy:")
print("{} correct in top 5, {} not in top 5. {.2f} top 5 accuracy.".format(
    correct, incorrect, (correct/(correct+incorrect))))

# Save model
model.save_model(overwrite=True)
