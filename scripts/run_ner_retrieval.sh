#!/bin/bash

PYTHONPATH=FEVEROUS/src python src/search/retrievers/run_retrieval.py --ner \
    --datadir $HOME/Data/FEVEROUS \
    --writedir $HOME/Data/FEVERRESULTS
