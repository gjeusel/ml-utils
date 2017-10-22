# -*- coding: utf-8 -*-

import sys
import os

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns


cmap_orrd = ListedColormap(sns.color_palette("OrRd", 10).as_hex())

def get_dist_mat(df, target_col, metric='euclidean', figsize=(20,20)):
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Sorting according to clusters to make then apparent :
    M = np.concatenate((X, y[:, np.newaxis]), axis=1)
    # Sort according to last column :
    M = M[M[:, -1].argsort()]
    M = M[0: -1]  # remove last column

    from scipy.spatial.distance import pdist, squareform
    dist_mat = pdist(M, metric=metric)
    dist_mat = squareform(dist_mat)  # translates this flattened form into a full matrix

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(dist_mat, cmap=cmap_orrd, interpolation='none')

    # get colorbar smaller than matrix
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # want a more natural, table-like display
    ax.invert_yaxis()

    # Move top xaxes :
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.axis('off')

    return dist_mat, fig, ax


def get_corr_mat(df, figsize=(20, 20)):
    # Compute correlation matrix
    corrmat = df.corr()

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set ax & colormap with seaborn.
    ax = sns.heatmap(corrmat, vmin=-1, vmax=1, center=0,
                     square=True, linewidths=1, xticklabels=True,
                     yticklabels=True)

    ax.set_xticklabels(df.columns, minor=False, rotation='vertical')
    ax.set_yticklabels(df.columns[df.shape[1]::-1], minor=False, rotation='horizontal')

    return corrmat, fig, ax


def boxplot(df, normalize=True, figsize=(20,20)):
    sns.set(style="ticks")
    # Initialize the figure with a logarithmic x axis
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        df = (df - df.mean()) / (df.max() - df.min())

    sns.boxplot(data=df,
                orient='h',
                # whis=whis # Proportion of the IQR past the low and high quartiles to extend the plot whiskers. Points outside this range will be identified as outliers.
                )

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)

    return fig, ax
