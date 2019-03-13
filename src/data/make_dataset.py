"""This is the script that takes the data output from the ChEMBL database
and cleans and processes it to create a mapping of SMILES to receptors"""
import os
import pandas as pd

from src.data.chembl_etl import ChEMBL_SQLite

PROJECT_BASE = "../../"
INTERIM_DATA = "data/interim/smiles_to_activity.csv"
FINAL_DATA = "data/processed/smiles_to_receptor.csv"


def main(basedir=""):
    """Currently this is a dummy script that doesn't do much data cleaning,
    this will be improved later"""

    if not os.path.isfile(basedir + INTERIM_DATA):
        chembl = ChEMBL_SQLite(path="data/")
        chembl.get_raw_data()

    data_set = pd.read_csv(basedir + INTERIM_DATA)

    data_set.drop(["published_value", "published_units"], axis=1, inplace=True)

    # Only save entries where we have more than 1000 data points per receptor
    y_classes = data_set["pref_name"].value_counts()
    y_classes = y_classes[y_classes >= 1000].index.tolist()
    data_set = data_set[~data_set["pref_name"].isin(y_classes)]

    data_set.to_csv(basedir + FINAL_DATA)


if __name__ == "__main__":
    main(basedir=PROJECT_BASE)
