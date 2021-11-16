import json

predicted = "/home/kw/Work/FEVEROUS/baseline_output/dev.combined.not_precomputed.p5.s5.t3.cells.verdict.jsonl"
# gold = "/home/kw/Data/FEVEROUS/dev.jsonl"

def sentence_table(evidence):
    """return (sentence evidence, table evidence)"""
    sentences = []
    tables = []
    for ev in evidence:
        if 'sentence' in ev:
            sentences.append(ev)
        elif 'cell' in ev:
            tables.append(ev)

    return sentences, tables


with open(predicted) as pred:
    jsonlines = pred.readlines()

    flag = False
    corr = incorr = 0

    for jsonline in jsonlines[1:]:
        line = json.loads(jsonline)

        claim = line["claim"]
        ev_gold = line["evidence"]
        ev_pred = line["predicted_evidence"]
        label_gold = line["label"]
        label_pred = line["predicted_label"]

        if label_gold != label_pred:
            flag = False
            # mismatching evidence
            for ev in ev_gold:
            # print(ev["content"])  
            # print(ev_pred)
            # print("--------------------------------")
                if set(ev["content"]).issubset(set(ev_pred)):
                    flag = True

            # if the thing is Wrong
            if not flag:
                incorr += 1
            else:
                corr += 1

    print(corr, incorr)

        # sent_pred, cells_pred = sentence_table(ev_pred)
        # print(sent_pred)
        # print(cells_pred)
