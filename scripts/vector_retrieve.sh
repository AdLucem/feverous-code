#!/bin/bash

echo "Running vector-based section retrieval on ADA"
PYTHONPATH=FEVEROUS/src python src/search/retrievers/run_retrieval.py --vector --split --N 20 --subN 20 \
    --datafile  /scratch/atreya/split_dev.pages.p20.jsonl \
    --indexdir /scratch/atreya/vector_index \
    --writefile /scratch/atreya/vector_retrieved_sections.jsonl \
    --db_path /scratch/atreya/feverous_wikiv1.db