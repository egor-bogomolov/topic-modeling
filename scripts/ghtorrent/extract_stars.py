import lzma
import sys

import bson  # from PyMongo


with lzma.open("repo_stars.txt.xz", "wb") as repo_stars:
    while True:
        try:
            for obj in bson.decode_file_iter(sys.stdin.buffer, codec_options=bson.CodecOptions(
                    unicode_decode_error_handler="ignore")):
                try:
                    obj
                    obj[0]["starred_url"]
                    obj["repo"]
                    obj["owner"]
                except KeyError:
                    continue
                repo_stars.write(f"{obj['owner']}/{obj['repo']} {len(obj)}".encode(errors="ignore"))
                repo_stars.write(b"\0")
                # commits.write(bytes.fromhex(obj["sha"]))
                # repos.write(obj["commit"]["url"][29:-53].encode())
                # repos.write(b"\0")
                # messages.write(obj["commit"]["message"].encode(errors="ignore"))
                # messages.write(b"\0")
            break
        except bson.errors.InvalidBSON:
            continue
