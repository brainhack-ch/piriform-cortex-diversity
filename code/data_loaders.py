from neuron import Neuron

import pandas as pd
import numpy as np
import xlrd

import os
import os.path as op

import ipywidgets


def get_correct_sheet_names(path_to_xls):
    wb = xlrd.open_workbook(path_to_xls)
    nb_sheets = len(wb.sheet_names())
    feat_names = []
    for s in range(nb_sheets):
        feat_names.append(wb.sheet_by_index(s).cell(0, 0).value)

    return feat_names


def genenerate_dendrite_df(excel_file, neuron_id, correct_sheet_names, verb=True):
    # Keeps only features starting with "Dendrite"
    dendrite_features = [
        feat for feat in correct_sheet_names if "Dendrite" in feat.split(" ")[0]
    ]

    if len(dendrite_features) < 1:
        if verb:
            print(f"Warning, no dendrite features in file {neuron_id} !")
        return pd.DataFrame([])

    # Convert first feature sheet into a dendrite DataFrame
    dendrite_df = excel_file.parse(
        correct_sheet_names.index(dendrite_features[0]), skiprows=1
    )  # \
    # .drop(columns = ['Unit', 'FilamentID', 'Time'])\
    # .loc[:, ['ID', 'Category', 'Depth', 'Level', 'Set 1', 'Dendrite Area']]

    col_to_keep = [
        col
        for col in dendrite_df.columns
        if col in ["ID", "Category", "Depth", "Level", "Set 1", "Dendrite Area"]
    ]
    dendrite_df = dendrite_df.loc[:, col_to_keep]

    if "Set 1" not in col_to_keep:
        print(
            f'Warning, no information about dendrite types (apical, etc) - setting "Set 1" column to "Unknown" !'
        )
        dendrite_df["Set 1"] = "Unknown"

    # Drop last empty row
    dendrite_df = dendrite_df.dropna(axis=0, subset=["ID"])

    # Convert ID to int (from scientific) - Warning
    dendrite_df["ID"] = dendrite_df.ID.astype(int).values
    # dendrite_df.loc[:, 'ID'] = dendrite_df.loc[:, 'ID'].astype(int).values

    # Iterate over features
    for feat in dendrite_features[1:]:
        sheet_nb = correct_sheet_names.index(feat)
        feat_df = excel_file.parse(sheet_name=sheet_nb, skiprows=1)  # \
        # .drop(columns = ['Unit', 'FilamentID', 'Time', 'Category', 'Depth', 'Level', 'Set 1'])\
        # .dropna(axis = 0, subset = ['ID'])

        col_to_drop = [
            col
            for col in feat_df.columns
            if col
            in ["Unit", "FilamentID", "Time", "Category", "Depth", "Level", "Set 1"]
        ]
        feat_df = feat_df.drop(columns=col_to_drop).dropna(axis=0, subset=["ID"])

        # removing "collection" columns for multi-data features
        if "Collection" in feat_df.columns:
            feat_df = feat_df.drop(columns=["Collection"])

        # Add new feature to existing dataframe
        dendrite_df = dendrite_df.merge(feat_df, on="ID", how="outer")

    if verb:
        print(
            f"Successfully converted data from {dendrite_df.shape[0]} dendrites ({dendrite_df.shape[1]} features found)."
        )

    return dendrite_df


def get_neuron_specific_features(
    excel_file,
    neuron_id,
    correct_sheet_names,
    features_list=[
        "Filament BoundingBoxOO Length A",
        "Filament BoundingBoxOO Length B",
        "Filament BoundingBoxOO Length C",
        "Filament Full Branch Depth",
        "Filament Full Branch Level",
        "Filament Length (sum)",
        "Filament No. Dendrite Branch Pts",
    ],
):
    # Initialize neuron features dict
    neuron_spec_features = {}

    # Iterate over specified features
    for feat in features_list:
        try:
            sheet_nb = correct_sheet_names.index(feat)
            feat_df = excel_file.parse(sheet_name=sheet_nb, skiprows=1).drop(
                columns=["Category", "Time", "ID"]
            )

        except ValueError:
            print(f'Feature "{feat}" could not be found for file {neuron_id}')
            feat_df = pd.DataFrame([])

        # Iterate over features to gather (if more than 1)
        cols_to_keeps = [
            col for col in feat_df.columns if col not in ["Unit", "Collection"]
        ]
        for col in cols_to_keeps:
            neuron_spec_features.update({col: feat_df.loc[0, col]})

    # Manually removes columns that could not be dropped
    # del neuron_spec_features['Unit']
    # del neuron_spec_features['Collection']

    return neuron_spec_features


def compute_AB_ratio(neuron_df, exclude = [], **kwargs):
    # Search for columns with "Basal" keyword
    columns_with_basal = [
        col
        for col
        in neuron_df.columns
        if "Basal"
        in col
        ]
    
    # Remove columns to excludes (e.g., branching angles)
    for excl in exclude:
        columns_with_basal = [
            col
            for col
            in columns_with_basal
            if excl
            not in col
            ]

    neuron_df_w_ratio = neuron_df.copy()

    for column_basal in columns_with_basal:
        column_apical = column_basal.replace("-Basal", "-Apical")
        ratio_name = column_basal.replace("-Basal", "-AB_ratio")

        # Compute the Apical/Basal ratio
        ratio = neuron_df.loc[:, column_apical]/neuron_df.loc[:, column_basal]
        neuron_df_w_ratio[ratio_name] = np.nan_to_num(ratio, posinf=0, neginf=0)

    return neuron_df_w_ratio


def excel_to_neuron(path_to_xls, verb=False, **kwargs):
    # Parse neuron id from path to excel file ('.' split removes file extension and ' ' removes potential ' - Copy')
    neuron_id = op.basename(path_to_xls).split(".")[0].split(" ")[0]
    if verb:
        print(f"Loading {neuron_id}")

    # Generate neuron
    new_neuron = Neuron(id=neuron_id, file_path=path_to_xls)

    neuron_file = pd.ExcelFile(path_to_xls)
    correct_sheet_names = get_correct_sheet_names(path_to_xls)

    if len(neuron_file.sheet_names) < 2:
        print(
            f"Warning, neuron {neuron_id} Excel file has only {len(neuron_file.sheet_names)} sheets ! \n it will be ignored."
        )
        return None

    # Gather neuron specific data
    neuron_features = get_neuron_specific_features(
        excel_file=neuron_file,
        neuron_id=neuron_id,
        correct_sheet_names=correct_sheet_names,
    )
    if len(neuron_features) < 1:
        print(f"Warning, no cell feature found in {neuron_id} file !")
    new_neuron.add_features(neuron_features)

    # Gather dendrite data in the form of a DataFrame
    dendrite_df = genenerate_dendrite_df(
        excel_file=neuron_file,
        neuron_id=neuron_id,
        correct_sheet_names=correct_sheet_names,
        verb=verb,
        **kwargs,
    )
    if dendrite_df.shape[1] < 1:
        print(f"Warning, no dendrite feature found in {neuron_id} file !")
    new_neuron.add_dendrites(dendrite_df)

    return new_neuron


def get_neuron_files_in_dir(dir_path):
    # xls_path = op.join(dir_path,'brain hack xls')
    files = os.listdir(dir_path)
    return [op.join(dir_path, f) for f in files if f.endswith("xls")]


def get_neuron_matrix(path_to_data, **kwargs):
    path = get_neuron_files_in_dir(path_to_data)

    # Neurons with no features are ignored
    first_neuron = None
    i = 0
    while first_neuron is None:
        first_neuron = excel_to_neuron(path[i], **kwargs)
        i += 1

    features_names = first_neuron.get_feature_dict().keys()

    n_cells = len(path)
    n_features = len(features_names)

    neuron_df = pd.DataFrame(np.zeros((n_cells, n_features)), columns=features_names)

    progressbar = ipywidgets.IntProgress(
        value=1,
        min=1,
        max=n_cells,
        description="Loading data:",
        bar_style="",
        style={"bar_color": "#AAAAAA"},
        orientation="horizontal",
    )
    display(progressbar)

    neuron_ids = []
    for i, neuron_path in enumerate(path):
        myneuron = excel_to_neuron(neuron_path, **kwargs)
        neuron_ids.append(myneuron.id)
        features = myneuron.get_feature_dict()
        neuron_df.loc[i] = features

        progressbar.value += 1

    progressbar.style = {"bar_color": "#81FF6B"}

    # Compute Apical/Basal ratio for selected features
    neuron_df_w_ratio = compute_AB_ratio(neuron_df)

    # Reset ID to match neuron specific ID
    neuron_df_w_ratio["id"] = neuron_ids
    neuron_df_w_ratio.set_index("id", inplace=True)

    return neuron_df_w_ratio.sort_index(axis=1)


def fill_neuron_layers(file_path, neuron_df):
    cell_info = pd.read_excel(file_path)
    cell_info.set_index("cell_id", inplace=True)
    updated_df = neuron_df.merge(
        cell_info, left_index=True, right_index=True, how="left"
    )
    updated_df["layer"].fillna(value="unknown", inplace=True)

    return updated_df
