#!/bin/bash
for f in *pdf; do
  gs -dSAFER -dBATCH -dNOPAUSE -r1200 -sDEVICE=tiff24nc -sCompression=lzw -dEPSCrop -sOutputFile="${f%.pdf}.tif" "$f"
done