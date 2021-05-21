import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


def normalize(v):
    return v / np.linalg.norm(v)


def cosine(v, u):
    return (normalize(v) * normalize(u)).sum()


def run_kmeans(n_clusters, embedding, init='random', normalize=False, random_seed: int = 42):
    if normalize:
        embedding = embedding.copy() / np.linalg.norm(embedding, axis=1, keepdims=True)
    clusterizer = KMeans(n_clusters, init=init, random_state=random_seed, verbose=1, n_init=5).fit(embedding)
    labels = clusterizer.predict(embedding)
    return clusterizer, labels


def build_clusterization(
        vec_file: str, token_file: str, counts_file: str, n_clusters: int,
        dendrogram_file: str = 'models/dendrogram_{}.pdf', pickle_file: str = 'models/clusterization_{}.pkl'
):
    embeddings = np.load(vec_file).reshape(-1, 100)
    print(embeddings.shape)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    tokens = [line.strip() for line in open(token_file, 'r')]
    tokens_counter = pickle.load(open(counts_file, 'rb'))

    kmeans, labels = run_kmeans(n_clusters, embeddings)

    cluster_labels = [str(i) for i in range(n_clusters)]

    for c in range(n_clusters):
        inds = np.where(labels == c)[0]
        sorted_by_freq = list(reversed(np.argsort([tokens_counter[tokens[ind]] for ind in inds])))

        print('~' * 20)
        print(f'Cluster {c}:')
        for ind in sorted_by_freq[:10]:
            true_ind = inds[ind]
            print(f'{tokens[true_ind]} {cosine(embeddings[true_ind], kmeans.cluster_centers_[c]):.3f}')
            cluster_labels[c] += " | " + tokens[true_ind]
        print('~' * 20)
        print()

    with open(pickle_file.format(n_clusters), 'wb') as fout:
        pickle.dump(kmeans, fout)
        pickle.dump(labels, fout)
        pickle.dump(cluster_labels, fout)

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
        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                          counts]).astype(float)
        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    X = kmeans.cluster_centers_
    n_clusters = len(X)

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='cosine', linkage='average')
    model = model.fit(X / np.linalg.norm(X, keepdims=True, axis=1))

    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    # plt.show()
    plt.figure(figsize=(10, max(10, n_clusters // 3)))
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    plot_dendrogram(model, truncate_mode='level', p=40, labels=cluster_labels, color_threshold=.8,
                    orientation='left', leaf_font_size=16)
    plt.savefig(dendrogram_file.format(n_clusters), bbox_inches='tight', dpi=100, format='pdf')


build_clusterization('models/vectors.npy', 'models/tokens.txt', 'models/token_counter.pkl', 32)
