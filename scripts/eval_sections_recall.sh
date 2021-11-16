#!/bin/bash

python src/evaluators/evaluate_sections_recall.py \
    --N 100 \
    --pred /scratch/atreya/vector_retrieved_sections.json \
    --gold /scratch/atreya/dev.jsonl