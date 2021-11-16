"""Evaluate coverage over top N pages retrieved for each claim- between the
retrieved pages and the gold pages in the evidence set"""

from os import listdir
from os.path import isfile, join
import json


def coverage(pages, evidence):
    """coverage:
    number of (gold) pages in predicted set/ total num_pages in gold set."""

    pred_pages = [page[0] for page in pages]
    gold_pages = list(set([ev.split("_")[0] for ev in evidence]))

    num_gold_in_retrieved = 0
    for glpg in gold_pages:
        if glpg in pred_pages:
            num_gold_in_retrieved += 1

    return num_gold_in_retrieved / len(gold_pages)


def average_coverage(filename, gold_filename):
    """Get average coverage of predicted pages over gold pages"""

    with open(filename) as f:
        predicted = list(map(lambda x: json.loads(x), f.readlines()))

    with open(gold_filename) as gf:
        gold = list(map(lambda x: json.loads(x), gf.readlines()))

    # for each prediction
    sum_coverage = 0
    for pred in predicted:
        id = pred["id"]
        pages = pred["predicted_pages"]

        # get gold prediction of same id
        gold_pred = [x for x in gold if x["id"] == id][0]
        evidences = gold_pred["evidence"]

        # Calculate coverage over all gold sets and take highest
        highest_coverage = 0
        for evidence in evidences:
            cv = coverage(pages, evidence["content"])
            if cv >= highest_coverage:
                highest_coverage = cv

        sum_coverage += highest_coverage

    avg = sum_coverage / len(predicted)
    return avg

# top-5    69.73%
# top-10   79.02%
# top-20   83.21%
# top-30   85.04%
# top-40   86.42%
# top-50   87.43%
# top-60   88.25%
# top-70   88.78%





# read in pages-retrieved-file
dirname = "/home/kw/Data/FEVEROUS/dev_pages"
gold_filename = "/home/kw/Data/FEVEROUS/dev.jsonl"

files = [f for f in listdir(dirname) if isfile(join(dirname, f))]
for filename in files:
    cov = average_coverage(join(dirname, filename), gold_filename)
    print("%20s %10f" % (filename, cov))


# "predicted_pages"
# read in annotated data

# get pages_retrieved for each gold sample

# match indexes

# compare coverage (???)