"""This is the script that takes the data output from the ChEMBL database
and cleans and processes it to create a mapping of SMILES to receptors"""
import os
import pandas as pd

from src.data.chembl_etl import ChEMBL_SQLite

PROJECT_BASE = "../../"
INTERIM_DATA = "data/interim/smiles_to_activity.csv"
FINAL_DATA = "data/processed/smiles_to_receptor.csv"


def main(basedir=""):
    """Run the SQL query, and do initial data processing and cleaning."""

    if not os.path.isfile(basedir + INTERIM_DATA):
        chembl = ChEMBL_SQLite()
        chembl.get_raw_data()

    data_set = pd.read_csv(basedir + INTERIM_DATA)

    # Drop records with <10mM
    data_set.drop(data_set[(data_set["standard_units"] == "mM") & (data_set["standard_value"] >= 10)].index, inplace=True)
    data_set.drop(data_set[(data_set["standard_units"] == "uM") & (data_set["standard_value"] >= 10000)].index, inplace=True)
    data_set.drop(data_set[(data_set["standard_units"] == "nM") & (data_set["standard_value"] >= 10000000)].index, inplace=True)

    # Drop binding information as we are done with it
    data_set.drop(["standard_value", "standard_units"], axis=1, inplace=True)

    print("Saving processed data")
    data_set.to_csv(basedir + FINAL_DATA)


if __name__ == "__main__":
    main(basedir=PROJECT_BASE)
