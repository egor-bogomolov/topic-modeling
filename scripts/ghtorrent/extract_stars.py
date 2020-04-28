import lzma
import sys
from pprint import pprint
from datetime import datetime
from collections import defaultdict

import bson  # from PyMongo

from tqdm import tqdm

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--repo_names", help="Names of interesting repositories")
args = parser.parse_args()


repo_data = {
    line.strip(): defaultdict(int)
    for line in open(args.repo_names, 'r').readlines()
}

repo_keys = [
    'forks_count', 'network_count', 'open_issues_count', 'language', 'size', 'stargazers_count', 'subscribers_count',
    'subscribers_count', 'updated_at'
]

issue_keys = [
    'issue_id', 'repo', 'owner', 'event'
]


def detect_obj_type(obj):
    is_repo = all([repo_key in obj] for repo_key in repo_keys) and ('full_name' in obj)
    if is_repo:
        return 'repo', obj['full_name']

    is_issue = all([issue_key in obj for issue_key in issue_keys])
    if is_issue:
        return 'issue', f'{obj["owner"]}/{obj["repo"]}'

    pprint(obj, indent=2)
    return None, None


while True:
    try:
        for obj in tqdm(bson.decode_file_iter(
                sys.stdin.buffer,
                codec_options=bson.CodecOptions(unicode_decode_error_handler="ignore")
        )):
            try:

                obj_type, name = detect_obj_type(obj)
                if obj_type is None or name not in repo_data:
                    continue

                if obj_type == 'repo':
                    repo = repo_data[name]
                    modification_date = datetime.strptime(obj['updated_at'], '%Y-%m-%dT%H:%M:%SZ')
                    obj['updated_at'] = int(modification_date.timestamp())
                    if repo['updated_at'] < obj['updated_at']:
                        for key in repo_keys:
                            repo[key] = obj[key]

                elif obj_type == 'issue':
                    repo = repo_data[name]
                    repo[f'issue_{obj["event"]}'] += 1

            except KeyError:
                continue
        break
    except bson.errors.InvalidBSON:
        continue

pprint(repo_data)

open('extracted_data.bson', 'wb').write(bson.encode(repo_data))
