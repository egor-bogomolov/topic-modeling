import os
import pickle
from collections import Counter, defaultdict

from joblib import Parallel, cpu_count, delayed
from dpu_utils.utils import RichPath
from tqdm import tqdm

from featurizer.data_loading import string_to_token_parts, aggregate_parts
from featurizer.utils import check_extension, author_modifications_dir, is_fork, AuthorResolver


def extract_counters_from_mods(repo_code_dir, author_ind, mods, fname):
    tokens = []
    for item in mods:
        hash_val = item['hash']
        path = item['path']
        line_inds = item['lines']
        timestamp = item['time']
        if check_extension(path):
            pre_dir = os.path.join(repo_code_dir, hash_val[:2])
            post_dir = os.path.join(pre_dir, hash_val[2:])
            path = os.path.join(post_dir, path.replace('/', '_').replace('\\', '_'))
            if not os.path.exists(path):
                continue
            all_lines = open(path, 'r').readlines()
            for ind in line_inds:
                if ind <= len(all_lines):
                    for token_parts in string_to_token_parts(all_lines[ind - 1].split()):
                        full_token = aggregate_parts(token_parts)
                        tokens.append((timestamp, full_token))

    output_dir = os.path.join('extracted_data', 'tokens_per_author', str(author_ind))
    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(tokens, open(os.path.join(output_dir, f'{fname}.pkl'), 'wb'))
    # return author_ind, tokens


author_resolver = AuthorResolver()

with Parallel(cpu_count() - 1) as pool:
    output_dir = os.path.join('extracted_data', 'tokens_per_author')
    os.makedirs(output_dir, exist_ok=True)

    pool(
        delayed(extract_counters_from_mods)(
            os.path.join('extracted_data', 'tokenized_code', f[:-len('.jsonl.gz')]),
            author_resolver.resolve(d['name'].lower(), d['email'].lower()),
            d['mods'],
            f[:-len(".jsonl.gz")]
        )
        for f in tqdm(os.listdir(author_modifications_dir))
        for d in RichPath.create(os.path.join(author_modifications_dir, f)).read_by_file_suffix()
        if not is_fork[f[:-len('.jsonl.gz')]]
    )

    # all_tokens = defaultdict(list)
    # for author, tokens in author_results:
    #     all_tokens[author].extend(tokens)
    #
    #
    # for author, tokens in all_tokens.items():
    #     tokens.sort()
    #     pickle.dump(tokens, open(os.path.join(output_dir, f'{author}_all_tokens.pkl'), 'wb'))
