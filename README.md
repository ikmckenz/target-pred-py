# [WIP] Small Molecule Target Predicition Py
This is a simple machine learning model to predict binding behavior of small molecule drugs in Python.

Similar work has been conducted by [SwissTargetPrediction](http://www.swisstargetprediction.ch/), [Predict NPS](https://www.predictnps.eu/), and [SuperPred](http://prediction.charite.de/).

The primary model is `StructureToMOAModel`, which predicts a mechanism of action from the structure of a drug-like molecule.
This model is trained by creating a data set of chemical structures (encoded as SMILES) mapped to mechanisms of action. 
The SMILES data is used to generate a feature vector for each molecule with chemical fingerprinting algorithms, and this is fed into a random forests machine learning algorithm.

This uses and includes data from ChEMBL, data is from http://www.ebi.ac.uk/chembl - the version of ChEMBL is
chembl_24_1.

Project based on the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) project template.