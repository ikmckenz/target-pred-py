# Small Molecule Target Predicition in Python
This is a simple machine learning model to predict binding behavior of small molecule drugs in Python.

Similar work has been conducted by [SwissTargetPrediction](http://www.swisstargetprediction.ch/), [Predict NPS](https://www.predictnps.com/), and [SuperPred](http://prediction.charite.de/).

### Model
Target Pred Py is a fairly simple model at present, but can be easily expanded an improved on.
Currently it uses FP6 fingerprints, and feeds them into a random forest classifier with a configurable number of trees. 
The sklearn random forest classifier holds all the decision trees in memory at the same time, and with the size of this data set (~200MB for just the features to SMILES with ChEMBL 25) the memory requirements increase rapidly along with the trees. 
It takes an AWS r5.4xlarge (with 128GB RAM) to train the model with 150 trees in the forest, and it would require roughly double the memory to serialize the model and save it for later use. 
Refactoring to a different random forest library, or writing our own, would help here. 

Currently the model with only FP6 fingerprints for features and only 150 trees in the random forest achieves 78% for top-1 precision, recall, and F1 score. 
Although increasing the number of trees from 10 (77% accuracy) to 150 (78% accuracy) provides minimal improvement, it provides a measurable difference in top-5 accuracy.
Top-5 accuracy increases in a linear fashion from 89% at 10 trees to 96% with 150. 
Adding more features from molecular descriptors or using an ensemble model would likely boost accuracy without much engineering effort. 
Also, experiments with different models such as Logistic Regression (like SwissTargetPrediction), SVMs, and neural networks should be tried.  

The primary model is in `StructureToMOAModel`, which predicts a mechanism of action from the structure of a drug-like molecule.
This model is trained by creating a data set of chemical structures (encoded as SMILES) mapped to mechanisms of action. 
The SMILES data is used to generate a feature vector for each molecule with chemical fingerprinting algorithms, and this is fed into a random forest machine learning algorithm.

### Getting started
To get up and running, first create the environment:
```bash
conda create --name target-pred-py
conda activate target-pred-py
conda install -c conda-forge -n target-pred-py rdkit textdistance
conda install -n target-pred-py scikit-learn
```
Then make the dataset, features, and train the model:
```bash
export PYTHONPATH=$PWD
cd src/data/
python make_dataset.py 
cd ../features/
python build_features.py
cd ../models/
python train_structuretomoa.py
```
Now you can predict on new molecules:
```bash
python predict_structuretomoa.py --smiles "yoursmilesstring"
```

### Comparisons 
* SwissTargetPrediction \[i] uses an ensemble of two models, one using shape similarities \[ii], the other using FP2 fingerprint similarity implemented by OpenBabel. 
Then they combine the two similarity measures using a multiple logistic regression.
    1. David Gfeller, Olivier Michielin, Vincent Zoete (2013) Shaping the interaction landscape of bioactive molecules.
    2. Armstrong, M.S.et al. (2011) Improving the accuracy of ultrafast ligand-basedscreening: incorporating lipophilicity into ElectroShape as an extra dimension.
* PredictNPS takes the Mold2 molecular descriptors and the FP6 fingerprint to create features, applies a variance and correlation filter, and then normalizes the data to create one feature vector. 
They feed this into a random forest classifier with 500 trees. 



Target Pred Py uses and includes data from ChEMBL, data is from http://www.ebi.ac.uk/chembl - the version of ChEMBL is
chembl_25.

Project structure based on the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) project template.