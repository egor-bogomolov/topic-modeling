import os
import pickle

from collections import defaultdict, Counter
from dpu_utils.utils import RichPath
from tqdm import tqdm

author_modifications_dir = os.path.join('extracted_data', 'author_modifications')

authors = set()
n2e = defaultdict(set)
e2n = defaultdict(set)

for f in tqdm(os.listdir(author_modifications_dir)):
    iterator = RichPath.create(os.path.join(author_modifications_dir, f)).read_by_file_suffix()
    for data in iterator:
        name = data['name'].lower()
        email = data['email'].lower()
        n2e[name].add(email)
        e2n[email].add(name)
        authors.add((name, email))

degE = Counter()
degN = Counter()


def useE(e):
    return degE[e] <= 5


def useN(n):
    return degN[n] < 5 or ' ' in n


for e, ns in e2n.items():
    degE[e] = len(ns)
for n, es in n2e.items():
    degN[n] = len(list(e for e in es if useE(e)))

colorsN, colorsE = {}, {}
colorsPairs = {}


def dfsN(n, c):
    colorsN[n] = c
    for e in n2e[n]:
        if useE(e) and e not in colorsE:
            dfsE(e, c)


def dfsE(e, c):
    colorsE[e] = c
    for n in e2n[e]:
        if useN(n) and n not in colorsN:
            dfsN(n, c)


color = 0
for n in n2e:
    if useN(n) and n not in colorsN:
        dfsN(n, color)
        color += 1

for e in e2n:
    if useE(e) and e not in colorsE:
        dfsE(e, color)
        color += 1

for name, email in authors:
    if name not in colorsN and email not in colorsE:
        colorsPairs[(name, email)] = color
        color += 1

with open('models/author_components.pkl', 'wb') as fout:
    pickle.dump(colorsE, fout)
    pickle.dump(colorsN, fout)
    pickle.dump(colorsPairs, fout)
