#!/bin/bash
# This script downloads the hg38 reference genome from the UCSC Genome Browser.

# Set the URL for the hg38 FASTA file
URL="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"

# Set the output directory
OUT_DIR="data/hg38"
mkdir -p $OUT_DIR

# Set the output file name
OUT_FILE="$OUT_DIR/hg38.fa.gz"

# Download the file
echo "Downloading hg38 from $URL..."
wget -O $OUT_FILE $URL

# Unzip the file
echo "Unzipping $OUT_FILE..."
gunzip $OUT_FILE

echo "Done."
