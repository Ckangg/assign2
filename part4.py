import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_hierarchical_cluster():
    
    return None

def fit_modified():
    return None


def compute():
    answers = {}
    random_state = 42

    # Generating the datasets
    noisy_circles = datasets.make_circles(n_samples=100, factor=.5, noise=.05, random_state=random_state)
    noisy_moons = datasets.make_moons(n_samples=100, noise=.05, random_state=random_state)
    varied = datasets.make_blobs(n_samples=100, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    aniso_data, aniso_labels = datasets.make_blobs(n_samples=100, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    aniso = (np.dot(aniso_data, transformation), aniso_labels)
    blobs = datasets.make_blobs(n_samples=100, random_state=random_state)
    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["4A: datasets"] = {"nc": [noisy_circles[0], noisy_circles[1]],
    "nm": [noisy_moons[0], noisy_moons[1]],
    "bvv": [varied[0], varied[1]],
    "add": [aniso[0], aniso[1]],
    "b": [blobs[0], blobs[1]]}
    
    def fit_hierarchical_cluster(data, linkage, n_clusters):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[0])
        hier_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        predicted_labels = hier_cluster.fit_predict(data_scaled)
        return predicted_labels

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """
    loaded_datasets = [noisy_circles, noisy_moons, varied, aniso, blobs]
    def fit_hier_cluster_scipy(data, method, n_clusters):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[0])
        linkage_method = 'average' if method == 'centroid' else method
        Z = scipy_linkage(data_scaled, method=linkage_method)
        predicted_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        return predicted_labels

# Apply hierarchical clustering with different linkage methods
    linkage_methods_scipy = ['single', 'complete', 'ward', 'centroid']
    hier_predictions_scipy_corrected = []

    for method in linkage_methods_scipy:
        predictions = [fit_hier_cluster_scipy(dataset, method, 2) for dataset in loaded_datasets]
        hier_predictions_scipy_corrected.append(predictions)

# Define the plotting function
    def plot_all_linkage_clusters(datasets, cluster_predictions, linkage_methods, title):
        num_linkages = len(linkage_methods)
        num_datasets = len(datasets)
        fig, axs = plt.subplots(num_linkages, num_datasets, figsize=(20, 16), sharex=True, sharey=True)
        fig.suptitle(title, fontsize=20)
    
        dataset_names = ['Noisy Circles', 'Noisy Moons', 'Varied', 'Anisotropic', 'Blobs']

        for i, linkage in enumerate(linkage_methods):
            for j, (dataset, pred_labels) in enumerate(zip(datasets, cluster_predictions[i])):
                axs[i, j].scatter(dataset[0][:, 0], dataset[0][:, 1], c=pred_labels, s=10, cmap='viridis')
                if i == num_linkages - 1:
                    axs[i, j].set_xlabel(dataset_names[j])
                if j == 0:
                    axs[i, j].set_ylabel(linkage)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Plot the results
    plot_all_linkage_clusters(loaded_datasets, hier_predictions_scipy_corrected, linkage_methods_scipy, 'Hierarchical Clustering with Different Linkages')
    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc","nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.
    
    Create a pdf of the plots and return in your report. 
    """
    def find_optimal_cutoff(Z):
    # Calculate the rate of change of distances between successive merges
        distances = Z[:, 2]
        distance_diff = np.diff(distances)
    
    # Find the maximum rate of change
        max_diff_idx = np.argmax(distance_diff)
        optimal_cutoff = distances[max_diff_idx]
    
        return optimal_cutoff

    def fit_hier_cluster_auto_cutoff(data, method):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[0])
        Z = scipy_linkage(data_scaled, method=method)

    # Find the optimal cutoff distance
        cutoff = find_optimal_cutoff(Z)
        predicted_labels = fcluster(Z, cutoff, criterion='distance') - 1  # Adjust labels to start from 0
    
        return predicted_labels

# Apply hierarchical clustering with automatic cutoff detection
    auto_cutoff_predictions = []
    for method in linkage_methods_scipy:
        predictions = [fit_hier_cluster_auto_cutoff(dataset, method) for dataset in loaded_datasets]
        auto_cutoff_predictions.append(predictions)

# Plot the results for hierarchical clustering with automatic cutoff detection
    plot_all_linkage_clusters(loaded_datasets, auto_cutoff_predictions, linkage_methods_scipy, 'Hierarchical Clustering with Automatic Cutoff')
    def fit_modified(data, method):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[0])
        Z = scipy_linkage(data_scaled, method=method)
    # Find the optimal cutoff distance
        cutoff = find_optimal_cutoff(Z)
        predicted_labels = fcluster(Z, cutoff, criterion='distance') - 1  # Adjust labels to start from 0
    return predicted_labels

    # dct is the function described above in 4.C
    dct = answers["4A: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
