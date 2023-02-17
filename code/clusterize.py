import numpy as np
import pandas as pd

#from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
#from scipy.cluster.hierarchy import dendrogram

#import matplotlib.pyplot as plt

import plotting_tools

def normalize(data):
    return (data - data.mean())/data.std()

def hierarchical_cstm(data, n_clusters = None, d_thresh = None, plot = True,
                        figsize = (35, 10), **kwargs):

    clustering = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=d_thresh, compute_full_tree=True, compute_distances=True).fit(data)
    labels = clustering.fit_predict(data)
    unique_labels = np.unique(labels)

    n_labels = labels.max() + 1

    cluster_distances = np.flip(clustering.distances_)[:n_labels+1]

    print('- Clustering stability for k clusters\n  ', ' '.join([f'k={i+2}: {dist:2.3f}' for i, dist in enumerate(np.abs(np.diff(cluster_distances)))]))

    labels_df = pd.DataFrame(labels, index = np.arange(len(labels)), columns = ['label'])

    if n_clusters is None:
        print(f'- Clustering yielded {n_labels} clusters')

    print('- Groups:', [np.sum(labels == i) for i in np.unique(labels)])

    label_to_color = None

    if plot:
        fig, ax = plotting_tools.initialize_dedrogram(d_thresh = d_thresh, n_labels = len(labels),
                                                        figsize = figsize)
        dendro_dict = plotting_tools.plot_dendrogram(clustering, color_threshold = d_thresh, above_threshold_color = 'gray', **kwargs)

        dendro_df = pd.DataFrame(dendro_dict['leaves_color_list'], index = dendro_dict['leaves'], columns = ['color'])
        label_to_color = labels_df.merge(dendro_df, left_index=True, right_index=True).groupby(by = 'label').first()

    return clustering, label_to_color

def get_centroids(cluster_model, dataframe, col, plot = False, 
                    figsize = (35, 10), fig = None, ax = None):
    
    labels = cluster_model.labels_
    unique_labels = np.unique(labels)

    #centroids = np.zeros((labels.max() + 1, dataframe.shape[1]))
    centroids = pd.DataFrame(index = np.arange(unique_labels.shape[0]), columns = dataframe.columns)

    for label_id in unique_labels:
        feat_in_label = dataframe.loc[labels == label_id].values
        centroids.loc[label_id] = feat_in_label.mean(axis = 0)

    if plot:
        fig, ax = plotting_tools.initialize_barplot()

        centroids_transposed = centroids.T
        centroids_transposed.columns = [f'cluster {i+1:02d}' for i in range(len(unique_labels))]
        centroids_transposed.plot.bar(ax = ax, color = col.color)

        ax.xaxis.grid(True)

    return centroids

def get_centroid_distance(cluster_model, dataframe, centroids, by_feature = None, how = 'mean'):

    labels = cluster_model.labels_
    unique_labels = np.unique(labels)

    feat_fltr = dataframe.columns.tolist()
    if type(by_feature) == str:
        feat_fltr = [by_feature]
    if type(by_feature) == list:
        feat_fltr = feat_fltr

    added_features = ['avg_radius', 'max_radius', 'feat_by_radius']

    #dist_to_centroid = centroids.loc[:, feat_fltr].copy()
    dist_to_centroid = pd.DataFrame(index = centroids.index, columns = added_features + centroids.columns.tolist())

    for label_id in unique_labels:

        # Computes distance from centroid per sample
        diff = (dataframe.loc[labels == label_id, feat_fltr] - centroids.loc[label_id]).astype(float)
        radius = np.linalg.norm(diff, axis = 1)

        radius_avg = radius.mean()
        radius_max = radius.max()

        # Computes distance by feature and sorting
        radius_by_feat = np.linalg.norm(diff, axis = 0, ord = 1)
        feat_sorted_by_rad = diff.columns[np.argsort(radius_by_feat)]

        dist_to_centroid.loc[label_id, 'avg_radius'] = radius_avg
        dist_to_centroid.loc[label_id, 'max_radius'] = radius_max
        dist_to_centroid.loc[label_id, 'feat_by_radius'] = feat_sorted_by_rad.tolist()
        dist_to_centroid.loc[label_id, centroids.columns.tolist()] = radius_by_feat

    return dist_to_centroid