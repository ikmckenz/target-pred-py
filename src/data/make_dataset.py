"""This is the script that takes the data output from the ChEMBL database
and cleans and processes it to create a mapping of SMILES to receptors"""
import os

import pandas as pd
import textdistance

from src.data.chembl_etl import ChEMBL_SQLite

PROJECT_BASE = "../../"
INTERIM_DATA = "data/interim/smiles_to_activity.csv"
FINAL_DATA = "data/processed/smiles_to_receptor.csv"
HAMMING_DISTANCE = 0.75


def main(basedir=""):
    """Run the SQL query, and do initial data processing and cleaning."""

    if not os.path.isfile(basedir + INTERIM_DATA):
        chembl = ChEMBL_SQLite()
        chembl.get_raw_data()

    data_set = pd.read_csv(basedir + INTERIM_DATA)
    print("Cleaning raw data")

    # Drop records with <10mM
    data_set.drop(data_set[(data_set["standard_units"] == "mM") & (data_set["standard_value"] >= 10)].index, inplace=True)
    data_set.drop(data_set[(data_set["standard_units"] == "uM") & (data_set["standard_value"] >= 10000)].index, inplace=True)
    data_set.drop(data_set[(data_set["standard_units"] == "nM") & (data_set["standard_value"] >= 10000000)].index, inplace=True)

    # Drop binding information as we are done with it
    data_set.drop(["standard_value", "standard_units"], axis=1, inplace=True)

    # Group targets by text similarity
    name_groupings = pd.DataFrame({"pref_name": data_set["pref_name"].unique()})
    name_groupings["group"] = -1
    n_names = name_groupings.shape[0]
    group = 0
    for row in range(n_names):
        if name_groupings.loc[row, "group"] == -1:
            name_groupings.loc[row, "group"] = group
            group += 1
            for row2 in range(row, n_names):
                score = textdistance.hamming.normalized_similarity(
                    name_groupings.loc[row, "pref_name"],
                    name_groupings.loc[row2, "pref_name"])
                if score > HAMMING_DISTANCE:
                    name_groupings.loc[row2, "group"] = name_groupings.loc[row, "group"]

    receptor_to_group_map = pd.Series(name_groupings["group"].values,
                                      index=name_groupings["pref_name"]).to_dict()
    group_to_agg_receptor = dict(
        name_groupings.groupby("group")["pref_name"].apply(lambda x: ", ".join(x)))
    data_set["pref_name"].replace(receptor_to_group_map, inplace=True)
    data_set["pref_name"].replace(group_to_agg_receptor, inplace=True)

    # Keep only rows where the target appears more than 100 times
    count = data_set["pref_name"].value_counts()
    data_set = data_set[data_set["pref_name"].isin(count.index[count > 100])]

    print("Saving processed data")
    data_set.to_csv(basedir + FINAL_DATA)


if __name__ == "__main__":
    main(basedir=PROJECT_BASE)
