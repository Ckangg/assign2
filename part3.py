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
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

"""
Part 3.	
Hierarchical Clustering: 
Recall from lecture that agglomerative hierarchical clustering is a greedy iterative scheme that creates clusters, i.e., distinct sets of indices of points, by gradually merging the sets based on some cluster dissimilarity (distance) measure. Since each iteration merges a set of indices there are at most n-1 mergers until the all the data points are merged into a single cluster (assuming n is the total points). This merging process of the sets of indices can be illustrated by a tree diagram called a dendrogram. Hence, agglomerative hierarchal clustering can be simply defined as a function that takes in a set of points and outputs the dendrogram.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def data_index_function(data, set_I, set_J):
    min_dist = np.inf  # Start with infinity as the minimum distance
    
    # Iterate over all pairs of points, one from set I and one from set J
    for i in set_I:
        for j in set_J:
            # Calculate the Euclidean distance between points i and j
            dist = np.linalg.norm(data[i] - data[j])
            # Update the minimum distance if this distance is smaller
            if dist < min_dist:
                min_dist = dist
    
    return min_dist
    return None


def compute():
    answers = {}

    """
    A.	Load the provided dataset “hierachal_toy_data.mat” using the scipy.io.loadmat function.
    """
    #file_path = "C:/Users/Ender CK/.vscode/all data/hierarchical_toy_data.mat"
    file_path="hierarchical_toy_data.mat"
    data = scipy.io.loadmat(file_path)
    answer1=data
    # return value of scipy.io.loadmat()
    answers["3A: toy data"] = answer1

    """
    B.	Create a linkage matrix Z, and plot a dendrogram using the scipy.hierarchy.linkage and scipy.hierachy.dendrogram functions, with “single” linkage.
    """
    X = data['X']

# Create the linkage matrix using single linkage
    Z = linkage(X, method='single')

# Plotting the dendrogram
    plt.figure(figsize=(10, 8))
    dendrogram(Z)
    answer2=dendrogram(Z)
    plt.title('Dendrogram with Single Linkage')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
    # Answer: NDArray
    answers["3B: linkage"] = np.zeros(1)

    # Answer: the return value of the dendogram function, dicitonary
    answers["3B: dendogram"] = answer2

    """
    C.	Consider the merger of the cluster corresponding to points with index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0) were these clusters merged? That is, what row does the merger of A correspond to in the linkage matrix Z? The rows count from 0. 
    """
    Z = linkage(X, method='single')
    set_1 = {8, 2, 13}
    set_2 = {1, 9}

    n = len(X)  # Number of initial data points

# Track which cluster each point belongs to at each step
    point_cluster = {i: {i} for i in range(n)}

# Iterate over the linkage matrix to see when the clusters merge
    for i, row in enumerate(Z):
        cluster_1, cluster_2 = int(row[0]), int(row[1])

    # Get the actual point sets for the clusters
        points_1 = point_cluster[cluster_1] if cluster_1 in point_cluster else set()
        points_2 = point_cluster[cluster_2] if cluster_2 in point_cluster else set()

    # Update the point clusters to reflect the merger
        new_cluster = points_1.union(points_2)
        for point in new_cluster:
            point_cluster[point] = new_cluster

    # Assign the new cluster to the higher index representing the merged cluster
            point_cluster[n + i] = new_cluster

    # Check if all points from set_1 and set_2 are now in the same cluster
            if set_1.issubset(new_cluster) and set_2.issubset(new_cluster):
                print(f"Clusters containing points {set_1} and {set_2} merged at iteration {i}.")
                break
    # Answer type: integer
    answers["3C: iteration"] = 4

    """
    D.	Write a function that takes the data and the two index sets {I,J} above, and returns the dissimilarity given by single link clustering using the Euclidian distance metric. The function should output the same value as the 3rd column of the row found in problem 2.C.
    """
    # Answer type: a function defined above
    answers["3D: function"] = data_index_function

    """
    E.	In the actual algorithm, deciding which clusters to merge should consider all of the available clusters at each iteration. List all the clusters as index sets, using a list of lists, 
    e.g., [{0,1,2},{3,4},{5},{6},…],  that were available when the two clusters in part 2.D were merged.
    """

    # List the clusters. the [{0,1,2}, {3,4}, {5}, {6}, ...] represents a list of lists.
    answers["3E: clusters"] = [{0}, {3}, {4}, {5}, {6}, {7}, {10}, {12}, {14}, {2, 13}, {9}, {1, 11}, {8}, set()]

    """
    F.	Single linked clustering is often criticized as producing clusters where “the rich get richer”, that is, where one cluster is continuously merging with all available points. Does your dendrogram illustrate this phenomenon?
    """

    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = "yes,The image suggests that there is a sequence where  smaller clusters are merged into a larger cluster step by step"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
