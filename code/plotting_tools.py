import numpy as np

from scipy.cluster.hierarchy import dendrogram

import matplotlib.pyplot as plt

def initialize_dedrogram(d_thresh, n_labels, d_line_mod = 11, figsize = (35, 10)):
    fig, ax = plt.subplots(figsize = figsize)

    plt.plot([5, d_line_mod*(n_labels - 1)], [d_thresh]*2, 'k:')

    return fig, ax

def initialize_barplot(figsize = (35, 10)):
    fig, ax = plt.subplots(figsize = figsize)

    return fig, ax

def save_figure(path_to_save, fig = None, ax = None):
    plt.tight_layout()
    plt.savefig(path_to_save, dpi = 300)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)