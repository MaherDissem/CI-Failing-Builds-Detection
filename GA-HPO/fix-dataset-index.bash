#!/bin/bash

cd "ordered-data"
for csv_file in ./*.csv
do
    echo "processing $csv_file"
    cut -d, -f1 --complement $csv_file > temp.csv && mv temp.csv $csv_file
done
