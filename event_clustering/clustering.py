import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# from https://www.kaggle.com/jbencina/clustering-documents-with-tfidf-and-kmeans
# plots the required plot to perform an elbow method for selecting the best cluster amount
def find_optimal_clusters(data, max_k):
    iters = range(2, max_k+1, 2)
    
    sse = []
    for k in iters:
        sse.append(MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')

# plots a PCA for the dataframe and given clusters
def plot_pca(df, clusters, path=''):
    pca = PCA(n_components=2).fit(df)
    coords = pca.transform(df)
    plt.figure()
    plt.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap='gnuplot')
    if len(path) > 0:
        plt.savefig(path)
    else:
        plt.show()
