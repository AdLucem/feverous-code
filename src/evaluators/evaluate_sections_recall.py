"""Evaluate recall@N over top N sections. this code probably will not make sense to anyone who is not the writer"""
# gold keys: 'id', 'claim', 'label', 'evidence', 'annotator_operations'
# predicted keys: claim', 'predicted_sections'

import json
import argparse

parser = argparse.ArgumentParser(description='Evaluate section recall over top N sections.')
parser.add_argument('--N', type=int, help='Top N sections')
parser.add_argument('--pred', type=str, help='File where predicted sections are stored in required format')
parser.add_argument('--gold', type=str, help='Gold data file')

args = parser.parse_args()

N = args.N
prediction_file = args.pred
gold_file = args.gold

# ====================== FUNCTIONS ===================


def get_last_section(context_ls):
    """Get the last element of the context list that is a section"""

    sections = [el for el in context_ls if "section_" in el]

    if sections == []:
        # unless last element is also title, in which case it's "introduction"
        last_section_id = "introduction"
    else:
        last_section = sections[-1]
        # section IDs in context list are of form, for eg:
        # "Algebraic logic_section_4". We want just the section ID
        last_section_id = "_".join(last_section.split("_")[1:])

    return last_section_id


def recall(predicted, gold):
    """Calculate recall over predicted and gold classes:
        Recall = n(TruePos) / (n(TruePos) + n(FalseNeg))
        Where
            TruePos: element in `predicted` that is in `gold`
            FalseNeg: element in `gold` that is not in `predicted`
    """

    true_pos = 0
    for element in predicted:
        if element in gold:
            true_pos += 1

    false_neg = 0
    for element in gold:
        if element not in predicted:
            false_neg += 1

    rc = true_pos / (true_pos + false_neg)
    # print(true_pos, false_neg)
    return rc


def recall_over_evidence_set(predicted, gold):

    # take top N elements of predicted sections only
    predicted_sections = predicted["predicted_sections"][:N]
    # get [(page_title, section_id)] list of predicted
    sections_pred = [sec.split("|") for sec in predicted_sections]

    # get [(page_title, section_id)] list of gold
    context_gold = gold['context']
    
    sections_gold = []
    # for each context list in the context dictionary
    for content_id in context_gold:
        context = context_gold[content_id]

        # the page title is first element of context list
        # without the "_title" suffix
        title_gold = context[0].split("_")[0]
        # get the gold section ID
        section_id_gold = get_last_section(context)

        sections_gold.append([title_gold, section_id_gold])

    # now calculate recall over predicted and gold
    rc = recall(sections_pred, sections_gold)

    return rc, sections_pred, sections_gold


def recall_for_sample(predicted, gold):
    """Get recall for the given predicted, gold samples"""

    # get all evidence sets for gold
    gold_evidence_sets = gold["evidence"]

    # take highest recall over all evidence sets
    top_rc = -1
    pred_set = []
    best_evidence_set = []
    for evidence_set in gold_evidence_sets:
        rc, sections_pred, sections_gold = recall_over_evidence_set(predicted, evidence_set)
        if rc > top_rc:
            top_rc = rc
            pred_set = sections_pred
            best_evidence_set = sections_gold

    # TEST PRINT
    # for title, sec_id in pred_set:
    #    print(title, sec_id)
    # print("------------------------------------")
    # for title, sec_id in best_evidence_set:
    #    print(title, sec_id)
    # print("------------------------------------")

    return pred_set, best_evidence_set, top_rc


# def test_print():
    
# ================= MAIN =========================


if __name__ == "__main__":

    # read in gold first
    with open(args.gold) as goldf:
        gold_data = [json.loads(line) for line in goldf.readlines()]

    # read in predicted dataset
    with open(prediction_file) as predf:
        pred_data = [json.loads(line) for line in predf.readlines()]

    # list of all recall values
    recall_ls = []

    # iterate through predicted data
    for prediction in pred_data:
        
        # get datapoint claim
        pred_claim = prediction["claim"]
          
        # get gold datapoint with matching claim
        gold = [dp for dp in gold_data if dp["claim"] == pred_claim][0]

        # get recall and also test print evidence
        pred_set, best_evset, rc = recall_for_sample(prediction, gold)

        print("===================================================")
        print(pred_claim)
        print("----------------------------------------------")
        for ev in best_evset:
            print(ev)
        print("----------------------------------------------")
        for ev in pred_set:
            print(ev)
        print("----------------------------------------------")
        print("Recall:", rc)

        recall_ls.append(rc)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Coverage @{n}".format(n=N))
    print("Coverage=1:", len([x for x in recall_ls if x == 1.0]))
    print("0<Coverage<1:", len([x for x in recall_ls if ((x != 1.0) and (x != 0.0))]))
    print("Coverage=0:", len([x for x in recall_ls if x == 0.0]))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
