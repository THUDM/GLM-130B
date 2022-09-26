#!/bin/bash
#
# NOTES
#
# - Must expand templates to avoid a large loss of content.
# - Text will not (redundantly) contain the title string.
# - Keep sections. Section title will be marked by "Section::::".
# - Keep lists. List bullets will be marked by "BULLET::::".
# - Keep tables. They're mostly garbage but can be removed later (remove "^!*").
# - Remove disambiguation pages. Right now there is no use for them.

INPUT=$1
PROCESSES=$2
TEMPLATES=$3
OUTPUT=$4

python WikiExtractor.py $INPUT \
       --json \
       --processes $PROCESSES \
       --templates $TEMPLATES \
       --output $OUTPUT \
       --bytes 1M \
       --compress \
       --links \
       --sections \
       --lists \
       --keep_tables \
       --min_text_length 0 \
       --filter_disambig_pages
