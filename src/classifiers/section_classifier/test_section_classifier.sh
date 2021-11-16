#!/bin/bash

./fetch_data.sh

python test_section_classifier.py \
    --model /scratch/atreya/feverous_model \
    --dev /scratch/atreya/dev_section_representation.jsonl \
    --lines 5000 \
    --save /scratch/atreya/


./export_data.sh

