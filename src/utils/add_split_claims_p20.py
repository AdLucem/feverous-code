import argparse  
import json
import re


parser = argparse.ArgumentParser(description='Include split claims in dev.pages.p20.jsonl -> split_dev.pages.p20.jsonl.')
parser.add_argument('--original', type=str, help='Original dev.pages.p20.jsonl')
parser.add_argument('--split', type=str, help='File of split claims, one line per claim')

args = parser.parse_args()


WRITE = "/scratch/atreya/split_dev.pages.p20.jsonl"

def split_on(delims, sentence):

    split = re.split(delims, sentence)
    split_ = [s for s in split if s.rstrip().lstrip() != ""]
    return split_


def run():

    with open(args.original) as forig:
        original_data = [json.loads(line) for line in forig.readlines()]

    with open(args.split) as fsplit:
        split_claims = fsplit.readlines()

    # original data json labels: evidence, id, claim, label, annotator_operations
    new_data = []
    for i, claim in enumerate(split_claims):

        # split on: <SEP> | , | . | ;
        claim_ = claim.rstrip("\n")
        split_claim = split_on("<SEP>|,|\.", claim_)

        new_subdata = original_data[i]
        new_subdata["split_claim"] = split_claim
        new_data.append(new_subdata)

    with open(WRITE, "w+") as wf:
        for p in new_data:
            wf.write(json.dumps(p) + "\n")    


if __name__ == "__main__":
    run()
