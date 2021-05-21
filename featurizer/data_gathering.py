import os
from tqdm import tqdm

with open('full_concatenation.txt', 'w') as fout:
    all_files = []
    for dirpath, dirnames, filenames in tqdm(os.walk('extracted_data/tokenized_code/'), total=752282):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            printed = False
            for line in open(full_path, 'r'):
                line = line.strip()
                if line:
                    if printed:
                        fout.write(' ')
                    fout.write(line)
                    printed = True
            if printed:
                fout.write('\n')
