# [WIP] Predict NPS Python
This is an implementation of the [Predict-NPS](https://www.predictnps.eu/) in-silico models to predict the behaviour and properties of New Psychoactive Substances (NPS) in Python.

The primary model is `StructureToMOAModel`, which predicts a mechanism of action from the structure of a drug-like molecule.
This model is trained by creating a data set of chemical structures (encoded as SMILES) mapped to mechanisms of action. 
The SMILES data is used to generate a feature vector for each molecule with chemical fingerprinting algorithms, and this is fed into a random forests machine learning algorithm.

This uses and includes data from ChEMBL, data is from http://www.ebi.ac.uk/chembl - the version of ChEMBL is
chembl_24_1.