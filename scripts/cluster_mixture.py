from argparse import ArgumentParser
from spherecluster import SphericalKMeans
import numpy as np
import pickle
from time import time

parser = ArgumentParser()
parser.add_argument("--n_clusters", help='Number of clusters', type=int, required=True)
parser.add_argument("--n_tokens", help='Use first N words', type=int)
parser.add_argument("--input", help='Path to .vec file produced by fasttext', type=str, required=True)
parser.add_argument("--output", help='Path to save pickle file', type=str, required=True)
parser.add_argument("--n_init", type=int, required=True)
parser.add_argument("--max_iter", type=int, required=True)
# parser.add_argument("--model", help='Choose between spherical k-means and MF', type=str)
args = parser.parse_args()

n_tokens = 0
dim = 0
words, vec = None, None

with open(args.input, 'r') as fin:
    print("Reading input data from {args.input}")

    n_tokens, dim = map(int, fin.readline().strip().split())
    if args.n_tokens is not None:
        n_tokens = args.n_tokens

    print(f"Using {n_tokens} with {dim} dimensions")

    words = np.empty(n_tokens, dtype=np.object)
    vec = np.empty((n_tokens, dim))

    for i in range(n_tokens):
        parts = fin.readline().strip().split()
        words[i] = parts[0]
        vec[i, :] = list(map(np.float, parts[1:]))

    print("Finished reading")

print("Fitting clusterization")
cur_time = time()

kmeans = SphericalKMeans(n_clusters=args.n_clusters, n_init=args.n_init, max_iter=args.max_iter, n_jobs=-1, verbose=1, random_state=42)
kmeans = kmeans.fit(vec)

print(f"Fitted in {time() - cur_time:.2f}")

pickle.dump(kmeans, open(args.output, "wb"))
print(f"Stored clusterizer in {args.output}")

