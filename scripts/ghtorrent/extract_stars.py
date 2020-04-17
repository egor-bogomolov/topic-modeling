import lzma
import sys
import bson  # from PyMongo

from collections import Counter

star_counts = Counter()

while True:
    try:
        for obj in bson.decode_file_iter(
                sys.stdin.buffer,
                codec_options=bson.CodecOptions(unicode_decode_error_handler="ignore")
        ):
            try:
                repo = f"{obj['owner']}/{obj['repo']}"
                star_counts[repo] += 1
            except KeyError:
                continue
        break
    except bson.errors.InvalidBSON:
        continue

with lzma.open("repo_stars.txt.xz", "wb") as repo_stars:
    for repo, count in star_counts.most_common():
        repo_stars.write(f"{repo} {count}\n".encode(errors="ignore"))
        repo_stars.write(b"\0")
