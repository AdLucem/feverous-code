#!/bin/bash

# echo "Running vector search indexing"
# PYTHONPATH=FEVEROUS/src python src/search/indexers/run_indexer.py --vector \
#    --datafile /home/shared/Data/feverous/FEVEROUS/dev.pages.p20.jsonl \
#    --index_dir /home/shared/Data/feverous/FEVEROUS/vector_index \
#    --db_path /home/shared/Data/feverous/FEVEROUS/feverous_wikiv1.db

echo "Running vector search indexing on ADA"
PYTHONPATH=FEVEROUS/src python src/search/indexers/run_indexer.py --vector \
    --datafile  /scratch/atreya/split_dev.pages.p20.jsonl \
    --index_dir /scratch/atreya/vector_index \
    --db_path   /scratch/atreya/feverous_wikiv1.db

