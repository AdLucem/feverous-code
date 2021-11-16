import argparse
import json

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage


parser = argparse.ArgumentParser()

parser.add_argument('--pages', type=str, help="Path to file containing top N pages for each claim")
parser.add_argument('--gold', type=str, help="Path to gold annotations file")
parser.add_argument('--db_path', type=str, help="Path to feverous_wikiv1.db")
parser.add_argument('--write', type=str, help="Path to data file to write to")


args = parser.parse_args()
print(args.pages)
print(args.gold)
print(args.db_path)
print(args.write)

# ==================== FUNCTIONS =======================


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


def get_page_sections_as_docs(db, title):
    """Get each section of the page as its own doc, in the format:
        {page title_section_id: text}"""

    page_json = db.get_doc_json(title)
    wiki_page = WikiPage(title, page_json)
    items = wiki_page.page_items

    # make {page_title|section_id: sentences} dict
    docs = {}
    cur_section = "introduction"
    cur_contents = []
    for item_key in items:

        # if section divider reacher
        if 'section' in item_key:
            # store sentences from previous section in docs
            docs[title + "_" + cur_section] = " ".join(cur_contents)

            # change section and contents to next
            cur_section = item_key
            cur_contents = []

        # else if content within a particular section
        else:

            # then add it to contents of current section
            element = wiki_page.get_element_by_id(item_key)
            cur_contents.append(element.__str__())

    # test print document set
    # for key in docs:
    #    print(key)
    #    print("=================================")
    #    for sent in docs[key]:
    #        print(sent)
    #        print("---------------------------------")
    #    print("+++++++++++++++++++++++++++++++++++++")
    return docs


# ======================= MAIN ============================

# get db
db = FeverousDB(args.db_path)

# keep gold annotations in RAM
with open(args.gold) as goldfile:
    gold_data = [json.loads(line) for line in goldfile.readlines()]

# print(getsizeof(gold_data) / 1024)

# open file to write to
writefile = open(args.write, "w+")

# read top N pages file line by line
with open(args.pages) as pagesfile:

    maybe_line = pagesfile.readline()
    i = 0

    while maybe_line != "":

        if i % 5000 == 0:
            print(i)
        i += 1

    # for i in range(4):
        data = json.loads(maybe_line)
        maybe_line = pagesfile.readline()

        claim_id = data["id"]
        claim = data["claim"]
        pages = data["predicted_pages"]
        pages = [pair[0] for pair in pages]

        section_docs = {}
        for page in pages:
            section_docs.update(get_page_sections_as_docs(db, page))
        predicted_evidence = list(section_docs.keys())
        
        # get gold evidence sets
        gold_evidences = [x for x in gold_data if x["id"] == claim_id][0]["evidence"]

        # join all gold evidence sets
        gold_evidence = []
        for evidence_set in gold_evidences:

            for evidence, context_ls in evidence_set["context"].items():
                page_title = evidence.split("_")[0]
                section_id = get_last_section(context_ls)
                gold_evidence.append(page_title + "_" + section_id)

        # now take gold_evidence sections out of predicted_evidence sections
        not_evidence = []

        for ev in predicted_evidence:
            if ev not in gold_evidence:
                not_evidence.append(ev)

        # make JSON
        writejson = {"id": claim_id,
                     "claim": claim,
                     "evidence": gold_evidence,
                     "not_evidence": not_evidence}

        # write it to file
        writefile.write(json.dumps(writejson) + "\n")

# close writefile
writefile.close()
