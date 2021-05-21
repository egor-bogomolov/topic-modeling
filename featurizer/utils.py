import os
import pickle

import dpu_utils
from dpu_utils.utils import save_jsonl_gz

author_modifications_dir = os.path.join('extracted_data', 'author_modifications')

pref = '../'
# pref = ''

is_fork = {}
for f in os.listdir(pref + 'repo_lists'):
    for data in dpu_utils.utils.RichPath.create(os.path.join(pref + 'repo_lists', f)).read_as_json():
        is_fork[data['name']] = data['fork']

repo_list = os.listdir(pref + 'repos')


class AuthorResolver:
    def __init__(self, fname=os.path.join(pref + 'models', 'author_components.pkl')):
        with open(fname, 'rb') as fin:
            self.colorsE = pickle.load(fin)
            self.colorsN = pickle.load(fin)
            self.colorsPairs = pickle.load(fin)
            self.missed = 0

    def resolve(self, name, email):
        name = name.lower()
        if name in self.colorsN:
            return self.colorsN[name]
        if name.replace(' ', '.') in self.colorsN:
            return self.colorsN[name.replace(' ', '.')]
        if name.replace('.', ' ') in self.colorsN:
            return self.colorsN[name.replace('.', ' ')]
        if email in self.colorsE:
            return self.colorsE[email]
        try:
            return self.colorsPairs[(name, email)]
        except KeyError:
            return None


def check_extension(fname):
    ext = fname.split('.')[-1]
    return ext.lower() in [
        "js", 'javascript', "py", "java", "go", "c", "cpp", "ruby", "rb",
        "ts", "tsx", "php", "cs", "sh", "zsh", "rs", "rust", "kotlin",
        "kt", "hs", "scala", "sc", "swift"
    ]
