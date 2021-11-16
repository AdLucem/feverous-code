# FEVEROUS Shared Task 

Work on the FEVEROUS shared task (in progress).

The FEVEROUS shared task includes two subtasks- evidence retrieval and NLI. We focus on the evidence retrieval subtask.

Since the search space is large (50GB), we first narrow the search space by retrieving most relevant articles and article sections.

## Most Relevant Articles

The corpus is narrowed by retrieving top N articles most relevant to each claim.

Recall@N:

```
N=5:  69.73%
N=10: 79.02%
N=20: 83.21%
N=30: 85.04%
N=40: 86.42%
N=50: 87.43%
N=60: 88.25%
N=70: 88.78%
```

## Most Relevant Sections

The corpus is further narrowed by retrieving top N most relevant sections (i.e: article subsections where evidence is most likely to be found.)

### NER-Based Retrieval

Retrieval using NER-based matching between claims and section text.

```bash
PYTHONPATH=FEVEROUS/src python src/search/retrievers/run_retrieval.py --ner --datadir <directory where feverous_wikiv1.db, dev.pages.p20.jsonl is stored> --writedir <directory to write results to>
```

Recall@20: 39% 

### Vector Similarity Retrieval

To make vector-based index for all documents in corpus:

```bash
PYTHONPATH=FEVEROUS/src python src/search/indexers/run_indexer.py --vector \
    --datafile <json file where claims data is stored> \
    --index_dir <directory to write index to> \
    --db_path <path to feverous_wikiv1.db>
```

To run retrieval with the above index:

```bash
PYTHONPATH=FEVEROUS/src python src/search/retrievers/run_retrieval.py --vector --N <top N sections> \
    --datafile <json file where claims are stored> \
    --indexdir <vector index generated in previous step> \
    --writefile <write to> \
    --db_path <path to feverous_wikiv1.db>
```

Recall@20: 37%
