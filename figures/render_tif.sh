#!/bin/bash
for f in *.eps; do
  gs -dSAFER -dBATCH -dNOPAUSE -r1200 -sDEVICE=tiff24nc -sCompression=lzw -dEPSCrop -sOutputFile="${f%.eps}.tif" "$f"
done