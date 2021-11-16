#!/bin/bash

python search/retrievers/BM25.py --corpus_dir /home/kw/Data/feverous_wikiv1_small.db --dataset /home/kw/Data/train_small.jsonl --k 10 --remove_stopwords true --section all 
