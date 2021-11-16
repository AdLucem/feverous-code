#!/bin/bash

PYTHONPATH=/home/kw/Work/FEVEROUS/src python representation_optimized.py \
    --train /home/kw/Data/FEVEROUS/sections_for_classification_tiny.jsonl \
    --write /home/kw/Data/FEVEROUS/section_representation_optim.jsonl \
    --db_path /home/kw/Data/FEVEROUS/feverous_wikiv1.db \
    --n 5 \
    --lines 1000 \
    --processes 4

# get section representations
# PYTHONPATH=/home/kw/Work/FEVEROUS/src python representation.py \
#    --train /home/kw/Data/FEVEROUS/sections_for_classification_tiny.jsonl \
#    --write /home/kw/Data/FEVEROUS/section_representation.jsonl \
#    --db_path /home/kw/Data/FEVEROUS/feverous_wikiv1.db \
#    --n 5 \
#    --lines 1000
# PYTHONPATH=/home/kw/Work/FEVEROUS/src python representation.py \
#    --train /scratch/atreya/sections_for_classification.jsonl \
#    --annotations /scratch/atreya/train.jsonl \
#    --db_path /home/kw/Data/FEVEROUS/feverous_wikiv1.db


# PYTHONPATH=/home/kw/Work/FEVEROUS/src python section_classifier.py \
#    --train /home/kw/Data/FEVEROUS/section_representation.jsonl \
#    --lines 4 \
#    --db_path /home/kw/Data/FEVEROUS/feverous_wikiv1.db \

# ADA version
# PYTHONPATH=../../FEVEROUS/src python section_classifier.py \
#    --train /scratch/atreya/section_representation.jsonl \
#    --lines 4 \
#    --db_path /scratch/atreya/feverous_wikiv1.db
