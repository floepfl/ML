# -*- coding: utf-8 -*-
"""Some plot functions."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from helper import build_distance_matrix


def plot_cluster(data, mu, colors, ax):
    """plot the cluster.

    Note that the dimension of the column vector `colors`
    should be the same as the number of clusters.
    """
    # check if the dimension matches.
    #assert(len(colors) >= mu.shape[0])
    # build distance matrix.
    distance_matrix = build_distance_matrix(data, mu)
    # get the assignments for each point.
    assignments = np.argmin(distance_matrix, axis=1)
    #
    for k_th in range(mu.shape[0]):
        rows = np.where(assignments == k_th)
        data_of_kth_cluster = data[rows, :]
        ax.scatter(
            np.squeeze(data_of_kth_cluster)[:, 0],
            np.squeeze(data_of_kth_cluster)[:, 1],
            s=40, c=colors[k_th % len(colors)])

    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot(data, mu, mu_old, out_dir):
    """plot."""
    colors = ['red', 'blue', 'green']
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    plot_cluster(data, mu_old, colors, ax1)
    ax1.scatter(mu_old[:, 0], mu_old[:, 1],
                facecolors='none', edgecolors='y', s=80)

    ax2 = fig.add_subplot(1, 2, 2)
    plot_cluster(data, mu, colors, ax2)
    ax2.scatter(mu[:, 0], mu[:, 1],
                facecolors='none', edgecolors='y', s=80)

    # matplotlib.rc('xtick', labelsize=5)
    # matplotlib.rc('ytick', labelsize=5)

    plt.tight_layout()
    plt.savefig(out_dir)
    plt.show()
    plt.close()

def plot_image_compression(original_image, processed_image, assignments, mu, k):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original_image, cmap='Greys_r')

    compressed_image = np.zeros(original_image.shape)

    # replace each pixel value by its cluster's value
    for k_th in range(k):
        indices_of_points_from_cluster_k = np.where(assignments == k_th)
        compressed_image[indices_of_points_from_cluster_k, :] = mu[k_th, :]

    compressed_image_reshaped = compressed_image.reshape(original_image.shape)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(compressed_image_reshaped, cmap='Greys_r')
    plt.draw()
    plt.pause(0.1)
    plt.tight_layout()
    plt.savefig("image_compression")
    plt.show()
