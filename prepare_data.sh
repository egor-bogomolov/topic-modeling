python3 load_files.py --flist data/gdrive/file_list_test.txt --output data_parts

cd data_parts
bzip2 -dk *.bz2
cd ..

#python3 load_files.py --flist data/gdrive/file_list_bow
