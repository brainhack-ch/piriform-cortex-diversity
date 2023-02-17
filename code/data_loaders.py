from neuron import Neuron

import pandas as pd
import numpy as np

import os
import os.path as op

import ipywidgets

def genenerate_dendrite_df(path_to_xls, verb = True):

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

    # Convert ID to int (from scientific) - Warning
    dendrite_df['ID'] = dendrite_df.ID.astype(int).values
    #dendrite_df.loc[:, 'ID'] = dendrite_df.loc[:, 'ID'].astype(int).values

    # Iterate over features
    for feat in dendrite_features[1:]:
        feat_df = pd.read_excel(path_to_xls, sheet_name = feat, skiprows = 1)\
                    .drop(columns = ['Unit', 'FilamentID', 'Time', 'Category', 'Depth', 'Level', 'Set 1'])\
                    .dropna(axis = 0, subset = ['ID'])
        
        # removing "collection" columns for multi-data features
        if 'Collection' in feat_df.columns:
            feat_df =  feat_df.drop(columns = ['Collection'])

        # Add new feature to existing dataframe
        dendrite_df = dendrite_df.merge(feat_df, on = 'ID', how = 'outer')
    
    if verb:
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

def excel_to_neuron(path_to_xls, **kwargs):

    # Parse neuron id from path to excel file ('.' split removes file extension and ' ' removes potential ' - Copy')
    neuron_id = op.basename(path_to_xls).split('.')[0].split(' ')[0]

    # Generate neuron
    new_neuron = Neuron(id = neuron_id, file_path = path_to_xls)

    # Gather neuron specific data
    neuron_features = get_neuron_specific_features(path_to_xls = path_to_xls)
    new_neuron.add_features(neuron_features)
 
    # Gather dendrite data in the form of a DataFrame
    dendrite_df = genenerate_dendrite_df(path_to_xls = path_to_xls, **kwargs)
    new_neuron.add_dendrites(dendrite_df)

    return new_neuron

def get_neuron_files_in_dir(dir_path):
    xls_path = op.join(dir_path,'brain hack xls')
    files = os.listdir(xls_path)
    return [op.join(xls_path,f) for f in files if f.endswith("xls")]

def get_neuron_matrix(dir_path, **kwargs):
    path = get_neuron_files_in_dir(dir_path)

    first_neuron = excel_to_neuron(path[0], **kwargs)
    features_names = first_neuron.get_feature_dict().keys()

    n_cells = len(path)
    n_features = len(features_names)

    neuron_df = pd.DataFrame(np.zeros((n_cells, n_features)), columns = features_names)
    
    progressbar = ipywidgets.IntProgress(value=1, min=1, max=n_cells,
                                             description='Loading data:', bar_style='',
                                             style={'bar_color': '#AAAAAA'},
                                             orientation='horizontal')
    display(progressbar)

    neuron_ids = []
    for i, neuron_path in enumerate(path):
        myneuron = excel_to_neuron(neuron_path, **kwargs)
        neuron_ids.append(myneuron.id)
        features = myneuron.get_feature_dict()
        neuron_df.loc[i]= features
        
        progressbar.value += 1
        
    neuron_df['id'] = neuron_ids
    neuron_df.set_index('id', inplace=True)
    return neuron_df

def fill_neuron_layers(file_path, neuron_df):
    cell_info = pd.read_excel(file_path)
    cell_info.set_index('cell_id',inplace=True)
    updated_df = neuron_df.merge(cell_info, left_index=True, right_index=True, how='left')
    updated_df['layer'].fillna(value='unknown', inplace=True)
    
    return updated_df
