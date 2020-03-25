import re
import os
import pickle

import numpy as np

from argparse import ArgumentParser
from collections import Counter
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm

pattern_repo = "\A\(('|\")(.*)('|\"),\s\["
pattern_words = "\(('|\")(\w+)('|\"), (\d+)\)"

parser = ArgumentParser()
parser.add_argument("--bow", help="Directory with unzipped BOW data", required=True)
args = parser.parse_args()


def extract_frequencies_from_file(lines):
    doc_frequency = Counter()
    term_frequency = Counter()
    # lines = open(fname, 'r').readlines()
    for line in lines:
        line = line.strip()
        word_counts = re.findall(pattern_words, line)
        for _, word, _, count in word_counts:
            doc_frequency[word] += 1
            term_frequency[word] += int(count)

    return doc_frequency, term_frequency


files = [f for f in os.listdir(args.bow) if not f.endswith('.lzo') and not f.endswith('.lzo.index')]
print(f'Loading data from {len(files)} files')
projects = [open(os.path.join(args.bow, fname), 'r').readlines() for fname in files]
n_documents = sum(map(len, projects))
print(f'Found {n_documents} projects')

doc_freq = Counter()
term_freq = Counter()

n_cpus = cpu_count()
with Parallel(n_cpus) as pool:
    print(f'Extracting stats from files')
    counters = pool([
        delayed(extract_frequencies_from_file)(project)
        for project in tqdm(projects)
    ])
    for d_counter, t_counter in counters:
        doc_freq += d_counter
        term_freq += t_counter

print(f'Computing idf and global tf-idf')
idf = Counter()
tf_idf = Counter()
for word, count in doc_freq.items():
    idf_word = np.log(n_documents / count)
    idf[word] = idf_word
    tf_idf[word] = term_freq[word] * idf_word

print('Dumping data')
pickle.dump(doc_freq, open('data/stats/doc_frequency.pkl', 'wb'))
pickle.dump(term_freq, open('data/stats/term_frequency.pkl', 'wb'))
pickle.dump(idf, open('data/stats/idf.pkl', 'wb'))
pickle.dump(tf_idf, open('data/stats/global_tf_idf.pkl', 'wb'))
