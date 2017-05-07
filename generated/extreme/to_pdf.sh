#!/usr/bin/env bash

for f in **/*.svg; do
    inkscape --file="$f" --export-pdf="$(dirname $f)/$(basename $f .svg).pdf"
done
