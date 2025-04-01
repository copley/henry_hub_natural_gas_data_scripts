#!/bin/bash
# This script splits a 6-page PDF into two documents:
# Pages 1-3 are saved as <original_filename>_ng.pdf
# Pages 4-6 are saved as <original_filename>_uo.pdf

# Check if an input file was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <pdf_file>"
    exit 1
fi

input_pdf="$1"

# Check if the input file exists
if [ ! -f "$input_pdf" ]; then
    echo "Error: File '$input_pdf' not found."
    exit 1
fi

# Remove the .pdf extension to create a base filename
base="${input_pdf%.pdf}"

# Construct output file names with the suffixes added before .pdf
output1="${base}_ng.pdf"
output2="${base}_uo.pdf"

# Split the PDF using pdftk
pdftk "$input_pdf" cat 1-3 output "$output1"
pdftk "$input_pdf" cat 4-6 output "$output2"

echo "Split complete:"
echo " - $output1 (pages 1-3)"
echo " - $output2 (pages 4-6)"

