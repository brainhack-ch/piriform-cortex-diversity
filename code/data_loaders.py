from neuron import Neuron

import pandas as pd
import numpy as np

import os
import os.path as op

import ipywidgets

def genenerate_dendrite_df(excel_file, neuron_id, verb = True):

    # Keeps only features starting with "Dendrite"
    dendrite_features = [feat for feat in excel_file.sheet_names if 'Dendrite' in feat.split(' ')[0]]

    if len(dendrite_features) < 1:
        if verb:
            print(f'Warning, no dendrite features in file {neuron_id} !')
        return pd.DataFrame([])
    
    
    # Convert first feature sheet into a dendrite DataFrame
    dendrite_df = excel_file.parse(dendrite_features[0], skiprows = 1)#\
                        #.drop(columns = ['Unit', 'FilamentID', 'Time'])\
                        #.loc[:, ['ID', 'Category', 'Depth', 'Level', 'Set 1', 'Dendrite Area']]
    
    col_to_keep = [col for col in dendrite_df.columns if col in ['ID', 'Category', 'Depth', 'Level', 'Set 1', 'Dendrite Area']]
    dendrite_df = dendrite_df.loc[:, col_to_keep]

    if 'Set 1' not in col_to_keep:
        print(f'Warning, no information about dendrite types (apical, etc) - setting "Set 1" column to "Unknown" !')
        dendrite_df['Set 1'] = 'Unknown'

    # Drop last empty row
    dendrite_df = dendrite_df.dropna(axis = 0, subset = ['ID'])

    # Convert ID to int (from scientific) - Warning
    dendrite_df['ID'] = dendrite_df.ID.astype(int).values
    #dendrite_df.loc[:, 'ID'] = dendrite_df.loc[:, 'ID'].astype(int).values

    # Iterate over features
    for feat in dendrite_features[1:]:
        feat_df = excel_file.parse(sheet_name = feat, skiprows = 1)#\
                    #.drop(columns = ['Unit', 'FilamentID', 'Time', 'Category', 'Depth', 'Level', 'Set 1'])\
                    #.dropna(axis = 0, subset = ['ID'])

        col_to_drop = [col for col in feat_df.columns if col in ['Unit', 'FilamentID', 'Time', 'Category', 'Depth', 'Level', 'Set 1']]
        feat_df = feat_df.drop(columns = col_to_drop)\
                        .dropna(axis = 0, subset = ['ID'])
        
        # removing "collection" columns for multi-data features
        if 'Collection' in feat_df.columns:
            feat_df =  feat_df.drop(columns = ['Collection'])

        # Add new feature to existing dataframe
        dendrite_df = dendrite_df.merge(feat_df, on = 'ID', how = 'outer')
    
    if verb:
        print(f'Successfully converted data from {dendrite_df.shape[0]} dendrites ({dendrite_df.shape[1]} features found).')

    return dendrite_df

def get_neuron_specific_features(excel_file, neuron_id, features_list = ['Filament BoundingBoxAA Length',
                                                                'Filament BoundingBoxOO Length',
                                                                'Filament Dendrite Length (sum)',
                                                                'Filament Distance from Origin']):

    # Initialize neuron features dict
    neuron_spec_features = {}

    # Iterate over specified features
    for feat in features_list:
        try:
            feat_df = excel_file.parse(sheet_name = feat, skiprows = 1)\
                        .drop(columns = ['Category', 'Time', 'ID'])
            
        except ValueError:
            print(f'Feature "{feat}" could not be found for file {neuron_id}')
            feat_df = pd.DataFrame([])

        # Iterate over features to gather (if more than 1)
        cols_to_keeps = [col for col in feat_df.columns if col not in ['Unit', 'Collection']]
        for col in cols_to_keeps:
            neuron_spec_features.update({col:feat_df.loc[0, col]})

    # Manually removes columns that could not be dropped
    #del neuron_spec_features['Unit']
    #del neuron_spec_features['Collection']

    return neuron_spec_features

def excel_to_neuron(path_to_xls, verb = False, **kwargs):

    # Parse neuron id from path to excel file ('.' split removes file extension and ' ' removes potential ' - Copy')
    neuron_id = op.basename(path_to_xls).split('.')[0].split(' ')[0]
    if verb:
        print(f'Loading {neuron_id}')

    # Generate neuron
    new_neuron = Neuron(id = neuron_id, file_path = path_to_xls)

    neuron_file = pd.ExcelFile(path_to_xls)

    if len(neuron_file.sheet_names) < 2:
        print(f'Warning, neuron {neuron_id} Excel file has only {len(neuron_file.sheet_names)} sheets !')

    # Gather neuron specific data
    neuron_features = get_neuron_specific_features(excel_file = neuron_file, neuron_id = neuron_id)
    if len(neuron_features) < 1:
        print(f'Warning, no cell feature found in {neuron_id} file !')
    new_neuron.add_features(neuron_features)
 
    # Gather dendrite data in the form of a DataFrame
    dendrite_df = genenerate_dendrite_df(excel_file = neuron_file, neuron_id = neuron_id, verb = verb, **kwargs)
    if dendrite_df.shape[1] < 1:
        print(f'Warning, no dendrite feature found in {neuron_id} file !')
    new_neuron.add_dendrites(dendrite_df)

    return new_neuron

def get_neuron_files_in_dir(dir_path):
    #xls_path = op.join(dir_path,'brain hack xls')
    files = os.listdir(dir_path)
    return [op.join(dir_path,f) for f in files if f.endswith("xls")]

def get_neuron_matrix(path_to_data, **kwargs):
    path = get_neuron_files_in_dir(path_to_data)

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
    
    progressbar.style = {'bar_color': '#81FF6B'}
        
    neuron_df['id'] = neuron_ids
    neuron_df.set_index('id', inplace=True)
    return neuron_df

def fill_neuron_layers(file_path, neuron_df):
    cell_info = pd.read_excel(file_path)
    cell_info.set_index('cell_id',inplace=True)
    updated_df = neuron_df.merge(cell_info, left_index=True, right_index=True, how='left')
    updated_df['layer'].fillna(value='unknown', inplace=True)
    
    return updated_df
