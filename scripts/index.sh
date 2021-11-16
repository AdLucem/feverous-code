#!/bin/bash

PYTHONPATH=FEVEROUS/src python src/search/indexers/tfidf_indexer.py \
    --top_pages /scratch/atreya/pages/dev.pages.p20.jsonl \
    --save_dir /scratch/atreya/dev \
    --db_path /scratch/atreya/feverous_wikiv1.db