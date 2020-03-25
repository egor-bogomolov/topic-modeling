import subprocess
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--flist", help="List of Google Drive files", type=str, required=True)
parser.add_argument("--output", help="Directory to save files", type=str, required=True)
args = parser.parse_args()

if not os.path.exists(args.output):
    os.mkdir(args.output)

with open(args.flist, 'r') as fin:
    lines = [l.strip().split() for l in fin.readlines()]
    for fname, fid in lines:
        subprocess.call(f"python3 download_google_drive/download_gdrive.py {fid} {os.path.join(args.output, fname)}", shell=True)

