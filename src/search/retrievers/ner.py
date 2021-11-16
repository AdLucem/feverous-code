"""Input: {claim, top 20 pages for each claim}
Step 1: Get NER(claim), NER(all section titles of top 20 pages)
Step 2: Retrieve all section titles with matching NER
Step 3: Get sections for claim (using section titles)
Step 4: Calculate (i) num_matching_sections, (ii) recall
"""
import json
from os.path import join
import spacy 
import numpy as np
from tqdm import tqdm

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage


def get_page_sections_as_docs(db, title):
    """Get each section of the page as its own doc, in the format:
        {page title|section_id: text"""

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
            docs[title + "|" + cur_section] = " ".join(cur_contents)

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


def make_into_docs(db, sample_pages, num_processes):
    """Make section-based docs for all pages in sample"""

    # all {doc_id: doc_text}
    docs = {}

    for page in sample_pages:
        # get each section of page as its own doc
        section_docs = get_page_sections_as_docs(db, page)
        docs.update(section_docs)

    # test-print docs
    # for doc in docs:
    #    print(doc)
    #    print("===================================")
    #    for sent in docs[doc]:
    #        print(sent)
    #    print("-----------------------------------")
    return docs


def show_ents(doc):
    """Display basic named entity info of spacy doc"""

    if doc.ents:
        for ent in doc.ents:
            print(ent.text, '-', str(ent.start_char),
                  '-', str(ent.end_char), '-', ent.label_)

    else:
        print("No named entities found")


def match(claim_doc, section_text):
    """Inputs: claim spacy doc
               section text
       Return: match score
    """

    # get list of entities for claim
    claim_ents = [ent.text for ent in claim_doc.ents]
    
    # score: number of claim_entities in section
    match_list = [a for a in claim_ents if a in section_text]

    return len(match_list)


def run(datadir, writedir):

    # top N sections
    N = 20 

    INPUTFILE = join(datadir, "dev.pages.p20.jsonl")
    WRITEFILE = join(writedir, "ner_predicted_sections.jsonl")
    db = FeverousDB(join(datadir, "feverous_wikiv1.db"))
    nlp = spacy.load('en_core_web_sm')

    with open(INPUTFILE) as f:
        data = [json.loads(line) for line in f.readlines()]
        
    # {claim: {section_title: section_content}}
    claim_sections = {}
    print("Converting sections to docs")

    for point in tqdm(data[:1000]):

        claim = point['claim']
        page_tuples = point["predicted_pages"]
        pages = [page[0] for page in page_tuples]

        docs = make_into_docs(db, pages, 4)
        claim_sections[claim] = docs
        
    # put both claims and sections into a flat list to make spacy-processing efficient
    all_claims = list(claim_sections.keys())

    # flattened list of section_ids
    all_section_ids = []
    all_section_texts = []
    for claim, sections in claim_sections.items():
        all_section_ids += sections.keys()
        all_section_texts += sections.values()

    # now put claims through spacy
    claim_spacy = []
    print("Processing claims through spacy")

    for cs in tqdm(nlp.pipe(all_claims, n_process=4)):
        claim_spacy.append(cs)

    # predicted data in format: [{claim_id, claim, predicted_sections}]
    print("Acquiring predicted sections for claim")
    PREDICTED = []
    for claim_index, claim in tqdm(enumerate(all_claims)):

        # section IDS for that specific claim
        section_ids = list(claim_sections[claim].keys())

        # indexes of section IDS
        section_ids_idx = [all_section_ids.index(secid) for secid in section_ids]

        # get docs corr. to indexes
        # and remove all corr. values from section IDs, section texts, section docs list
        section_docs = []
        for idx in section_ids_idx:
            section_docs.append(all_section_texts[idx])

        # get matches for all section docs with that claim
        claim_doc = claim_spacy[claim_index]
        match_scores = [match(claim_doc, sec_doc) for sec_doc in section_docs]
        
        # take indexes of top 20 match scores
        top_N_idx = np.argsort(match_scores)[-20:]
        
        # get section IDs corresponding to top_N_idx
        top_N_section_ids = [section_ids[i] for i in top_N_idx]

        subdata = {"claim_id": claim_index,
                   "claim": claim,
                   "predicted_sections": top_N_section_ids}
        PREDICTED.append(json.dumps(subdata) + "\n")

    with open(WRITEFILE, "w+") as wf:
        for line in PREDICTED:
            wf.write(line)
