import os
import pickle
from argparse import ArgumentParser

import numpy as np

from featurizer.utils import AuthorResolver


def build_vectors(developers_file: str, n_clusters: int, token_file: str):
    developers = [line.strip() for line in open(developers_file, 'r')]
    resolver = AuthorResolver()
    components = []
    for dev in developers:
        try:
            components.append((resolver.resolve(dev, ''), dev))
        except KeyError:
            print(f"Missing {dev}")

    tokens = [line.strip() for line in open(token_file, 'r')]
    with open(f'models/clusterization_{n_clusters}.pkl', 'rb') as fin:
        kmeans = pickle.load(fin)
        labels = pickle.load(fin)
        cluster_labels = pickle.load(fin)

    # timestamps = np.load('models/issue_timestamps.npy')
    timestamps = np.array(list(pickle.load(open('issues/data/ts_map.pkl', 'rb')).values()))
    np.sort(timestamps)

    token_to_label = {token: label for token, label in zip(tokens, labels)}

    all_data = {}
    for dev_c, dev in components:
        dev_dir = f'extracted_data/tokens_per_author/{dev_c}'
        dev_clusters = np.zeros(n_clusters)
        try:
            for f in os.listdir(dev_dir):
                developer_data = pickle.load(open(os.path.join(dev_dir, f), 'rb'))
                current_ind = 0
                for timestamp in timestamps:
                    while current_ind < len(developer_data) and developer_data[current_ind][0] <= timestamp:
                        dev_token = developer_data[current_ind][1]
                        if dev_token in token_to_label:
                            dev_clusters[token_to_label[dev_token]] += 1
                        current_ind += 1
                    all_data[(dev_c, int(timestamp))] = dev_clusters.copy()
        except FileNotFoundError:
            continue

        # print('=' * 20)
        # print(dev, ':')
        # for cluster_ind in list(reversed(np.argsort(dev_clusters)))[:20]:
        #     print(dev_clusters[cluster_ind], '|', cluster_labels[cluster_ind])
        # print('=' * 20)

    dev_out_dir = f'extracted_data/vectors_per_author'
    os.makedirs(dev_out_dir, exist_ok=True)
    pickle.dump(all_data, open(os.path.join(dev_out_dir, f'clusters_{n_clusters}.pkl'), 'wb'))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-d', type=str, required=True)
    arg_parser.add_argument('-c', type=int, required=True)
    arg_parser.add_argument('-t', type=str, required=True)
    args = arg_parser.parse_args()

    build_vectors(args.d, args.c, args.t)
