#!/bin/bash

./fetch_data.sh

python section_classifier.py \
    --train /scratch/atreya/section_representation_medium.jsonl \
    --save_model /scratch/atreya/feverous_model_chungus_test \
    --lines 100

./export_data.sh


# get section representations
# PYTHONPATH=/home/kw/Work/FEVEROUS/src python representation_optimized.py \
#    --train /home/kw/Data/FEVEROUS/sections_for_classification_tiny.jsonl \
#    --write /home/kw/Data/FEVEROUS/section_representation_optim.jsonl \
#    --db_path /home/kw/Data/FEVEROUS/feverous_wikiv1.db \
#    --n 5 \
#    --lines 100

# non-optimized but working version
# PYTHONPATH=/home/kw/Work/FEVEROUS/src python representation.py \
#    --train /home/kw/Data/FEVEROUS/dev_sections_for_classification.jsonl \
#    --write /home/kw/Data/FEVEROUS/dev_section_representation.jsonl \
#    --db_path /home/kw/Data/FEVEROUS/feverous_wikiv1.db \
#    --lines 1000

#PYTHONPATH=/home/kw/Work/FEVEROUS/src python classifiers/section_classifier.py --train /home/kw/Data/FEVEROUS/sections_for_classification.jsonl

# PYTHONPATH=/home/kw/Work/FEVEROUS/src python section_classifier.py \
#    --train /home/kw/Data/FEVEROUS/section_representation.jsonl \
#    --lines 4 \
#    --db_path /home/kw/Data/FEVEROUS/feverous_wikiv1.db \

# ADA version
# PYTHONPATH=../../FEVEROUS/src python section_classifier.py \
#    --train /scratch/atreya/section_representation.jsonl \
#    --lines 4 \
#    --db_path /scratch/atreya/feverous_wikiv1.db
