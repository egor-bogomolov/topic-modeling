import pickle

from tqdm import tqdm
from .data_loading import string_to_token_parts, aggregate_parts
from collections import Counter

token_counter = Counter()

for line in tqdm(open('full_concatenation.txt', 'r')):
    for token_parts in string_to_token_parts(line.split()):
        full_token = aggregate_parts(token_parts)
        token_counter[full_token] += 1

print(token_counter.most_common(200))

pickle.dump(token_counter, open('models/token_counter.pkl', 'wb'))
