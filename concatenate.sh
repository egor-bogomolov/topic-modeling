IFS=$'\n'
i=1
for path in `find extracted_data/tokenized_code/ -type f`;
do
	if [ $(($i%1000)) == 0 ]; then
		echo $i/4129226
	fi
	#filename=$(basename -- "$path")
	#extension="${filename##*.}"
	cat "$path" >> extracted_data/concatenated_data/concatenated_nightly.txt
	let "i=i+1"
done
