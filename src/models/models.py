"""The machine learning models"""

import os
import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import TensorDataset, DataLoader


class StructureToMOAModel(ABC):
    """Abstract class for the models which take in a drug structure and output a predicted
    mechanism of action"""

    def __init__(self,
                 y_transform,
                 model_save_path,
                 project_base="../../"):

        self.y_transform = y_transform

        self.project_base = project_base
        self.model_save_path = model_save_path
        self.model = None
        self.trained = False

    @abstractmethod
    def train(self, X, y):
        """Train the model

        Args:
            X: training input samples
            y: training target values

        """

    @abstractmethod
    def predict(self, data):
        """Predict a MOA from a structure

        Args:
            data: The structure to predict

        """

    @abstractmethod
    def predict_top(self, data, n_outputs):
        """Predict top n MOAs from a structure

        Args:
            data: The structure to predict
            n_outputs: The number of predictions

        Returns:
            top: list of numeric category predictions
            probabilities: list of probabilities
        """

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

    @abstractmethod
    def load_model(self):
        """Load a saved model"""

    @abstractmethod
    def _save_model(self, file_loc):
        """Private function to pickle the model"""


class StructureToMOARFModel(StructureToMOAModel):
    """Defines the random forest classifier model from structure to MOA"""

    def __init__(self,
                 y_transform=None,
                 project_base="../../",
                 model_save_path="models/structuretomoa_model.pickle",
                 n_estimators=10):
        StructureToMOAModel.__init__(self, y_transform=y_transform,
                                     model_save_path=model_save_path,
                                     project_base=project_base)

        self.model = RandomForestClassifier(n_estimators=n_estimators,
                                            n_jobs=-1)

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, data):
        return self.model.predict(data)

    def predict_top(self, data, n_outputs=5):
        model_output = self.model.predict_proba(data)
        top = np.argsort(model_output)[:, ::-1][:, :n_outputs]
        probabilities = np.take_along_axis(model_output, top, 1)
        return top, probabilities

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
            pickle.dump([self.model, self.y_transform], f, protocol=4)


class StructureToMOANNModel(StructureToMOAModel):
    """Implements the Structure to MOA model using a neural network"""

    def __init__(self,
                 y_transform=None,
                 project_base="../../",
                 model_save_path="models/nn_model.pickle",
                 device=None,
                 n_classes=120,
                 n_features=2048):
        StructureToMOAModel.__init__(self, y_transform=y_transform,
                                     model_save_path=model_save_path,
                                     project_base=project_base)
        self.model = self.Net(n_classes=n_classes, n_features=n_features)
        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(self.device)

    class Net(nn.Module):
        """The neural net itself, with two hidden layers"""
        def __init__(self, n_classes=120, n_features=2048):
            super().__init__()

            self.l1 = nn.Linear(n_features, 1024)
            self.l2 = nn.Linear(1024, 512)
            self.output = nn.Linear(512, n_classes)

        def forward(self, x):
            """The forward pass of the neural net, pass the input tensor through each of our
             operations"""
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = self.output(x)

            return x

    def train(self, X, y):
        import time
        start = time.time()
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X).float()
        if not isinstance(y, torch.Tensor):
            y = torch.Tensor(y).long()
        print("Training on {}".format(self.device))
        my_dataset = TensorDataset(X, y)
        trainloader = DataLoader(my_dataset, batch_size=100, shuffle=True, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)

        for epoch in range(30):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        self.trained = True
        end = time.time()
        print(start - end)

    def predict(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(data)
            _, predicted = torch.max(outputs.data, 1)
            np_pred = predicted.detach().cpu().numpy()

        return np_pred

    def predict_top(self, data, n_outputs=5):
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data).float().to(self.device)

        with torch.no_grad():
            outputs = self.model(data)
            prob, predicted = torch.topk(outputs.data, 5, 1)
            np_pred = predicted.detach().cpu().numpy()
            np_prob = prob.detach().cpu().numpy()

        return np_pred, np_prob

    def load_model(self):
        file_loc = self.project_base + self.model_save_path
        if os.path.isfile(file_loc):
            with open(file_loc, "rb") as f:
                data = pickle.load(f)
                self.model.load_state_dict(data[0])
                self.y_transform = data[1]
        else:
            print("Could not find model to load")

    def _save_model(self, file_loc):
        with open(file_loc, 'wb') as f:
            pickle.dump([self.model.state_dict(), self.y_transform], f, protocol=4)
