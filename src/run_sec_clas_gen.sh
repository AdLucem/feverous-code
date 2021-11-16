#!/bin/bash

PYTHONPATH=/home/kw/Work/FEVEROUS/src \
    python analyze_data/generate_section_classification_data.py \
    --pages /home/kw/Data/FEVEROUS/dev.pages.p20.jsonl \
    --gold /home/kw/Data/FEVEROUS/dev.jsonl \
    --db_path /home/kw/Data/FEVEROUS/feverous_wikiv1.db \
    --write /home/kw/Data/FEVEROUS/dev_sections_for_classification.jsonl
