from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.features.build_features import Features
from src.models.models import StructureToMOAModel

# Training data
print("Loading training features")
features = Features()
X, y, y_transform = features.load_training_features()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Fitting model
print("Fitting model")
model = StructureToMOAModel(y_transform=y_transform)
model.train(X_train, y_train)

# Predicting test set
print("Predicting test set")
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save model
model.save_model(overwrite=True)
