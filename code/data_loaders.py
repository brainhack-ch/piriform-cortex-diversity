from neuron import Neuron

import pandas as pd
import os.path as op
import os

def genenerate_dendrite_df(path_to_xls):

    # Load neuron excel file
    neuron_excel = pd.ExcelFile(path_to_xls)

    # Keeps only features starting with "Dendrite"
    dendrite_features = [feat for feat in neuron_excel.sheet_names if 'Dendrite' in feat.split(' ')[0]]

    # Convert first feature sheet into a dendrite DataFrame
    dendrite_df = pd.read_excel(path_to_xls, sheet_name = dendrite_features[0], skiprows=1)\
                        .drop(columns = ['Unit', 'FilamentID', 'Time'])\
                        .loc[:, ['ID', 'Category', 'Depth', 'Level', 'Set 1', 'Dendrite Area']]

    # Drop last empty row
    dendrite_df = dendrite_df.dropna(axis = 0, subset = ['ID'])

    # Convert ID to int (from scientific)
    dendrite_df.loc[:, 'ID'] = dendrite_df.ID.astype(int)

    # Iterate over features
    for feat in dendrite_features[1:]:
        feat_df = pd.read_excel(path_to_xls, sheet_name = feat, skiprows = 1)\
                    .drop(columns = ['Unit', 'FilamentID', 'Time', 'Category', 'Depth', 'Level', 'Set 1'])\
                    .dropna(axis = 0, subset = ['ID'])

        # Add new feature to existing dataframe
        dendrite_df = dendrite_df.merge(feat_df, on = 'ID', how = 'outer')

    print(f'Successfully converted data from {dendrite_df.shape[0]} dendrites ({dendrite_df.shape[1]} features found).')

    return dendrite_df

def get_neuron_specific_features(path_to_xls, features_list = ['Filament BoundingBoxAA Length',
                                                                'Filament BoundingBoxOO Length',
                                                                'Filament Dendrite Length (sum)',
                                                                'Filament Distance from Origin']):

    # Initialize neuron features dict
    neuron_spec_features = {}

    # Iterate over specified features
    for feat in features_list:
        feat_df = pd.read_excel(path_to_xls, sheet_name = feat, skiprows = 1)\
                    .drop(columns = ['Category', 'Time', 'ID'])

        # Iterate over features to gather (if more than 1)
        for col in feat_df.columns:
            neuron_spec_features.update({col:feat_df.loc[0, col]})

    # Manually removes columns that could not be dropped
    del neuron_spec_features['Unit']
    del neuron_spec_features['Collection']

    return neuron_spec_features

def excel_to_neuron(path_to_xls):

    # Parse neuron id from path to excel file ('.' split removes file extension and ' ' removes potential ' - Copy')
    neuron_id = op.basename(path_to_xls).split('.')[0].split(' ')[0]

    # Generate neuron
    new_neuron = Neuron(id = neuron_id, file_path = path_to_xls)

    # Gather neuron specific data
    neuron_features = get_neuron_specific_features(path_to_xls = path_to_xls)
    new_neuron.add_features(neuron_features)
 
    # Gather dendrite data in the form of a DataFrame
    dendrite_df = genenerate_dendrite_df(path_to_xls = path_to_xls)
    new_neuron.add_dendrites(dendrite_df)

    return new_neuron

def get_neuron_files_in_dir(dir_path):
    xls_path = op.join(dir_path,'brain hack xls')
    files = os.listdir(xls_path)
    return [op.join(xls_path,f) for f in files if f.endswith("xls")]