"""Representation of each section/claim/label triplet as a sequence that can be fed to RoBERTa

- step 1: 
    convert all the data into a {claim|page_id_section_id|sentence_id: sentence} sequence. Same for table cells and other items
    - step 1.5: make a separate dict with 
                {claim_id|page_id_section_id: label}

- step 2: 
    make [id], [sentence] lists for dict in step 1. Preprocess all items in [sentence] to prepare for TF-IDF vectorization, and make a new list corresponding to [ids]:
                [preproc_sentences]

- step 3: 
    tf-idf vectorize list from step 3, and get matrix `tfidf_matrix`

- step 4:
    get claims_sections := {claim|section_id: [indexes]} set out of `ids`

- step 5:
    for claim_section in claims_sections:
        - 5.1: get cosine_sim values for all sentences in c_s
        - 5.2: get top `n` sentences using cosine_sims
          
- Step 6: 
    get top `n` sequences out of top_n_sentences
    where
        top_n_sequences = [(sentence_before, sentence, sentence_after) for sentence in top_n_sentences]

Output of file:
           [{claim, 
            {sectionid, [sequences], label}}] 
"""

import argparse
import json
from tqdm import tqdm
import spacy
import numpy as np

from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, load_npz, csr_matrix


parser = argparse.ArgumentParser()

parser.add_argument('--train', type=str, help="Path to training data file")
parser.add_argument('--write', type=str, help="Path to file to write to")
parser.add_argument('--db_path', type=str, help="Path to feverous_wikiv1.db")
parser.add_argument('--n', type=int, help="Number of sentences per section")
parser.add_argument('--lines', type=int, help="Number of lines in training data file")
parser.add_argument('--processes', type=int, help="Number of processes to run spacy's pipe on", default=4)
args = parser.parse_args()


# ====================== FUNCTIONS =========================


def preprocess(doc):
    """Stems all non-stopword words in sentence and then puts words
    back together- whitespace-separated- again"""

    stemmer = PorterStemmer()
    stopword_list = stopwords.words('english')
    tokens = word_tokenize(doc)

    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stopword_list]
    
    return " ".join(stemmed_tokens)


def section_off_page(items):
    """Input: ordered list of items in page
       Output: {section: [items]}"""

    elements = {}
    cur_items = []
    cur_section = "introduction"

    # iterate through items
    for item in items:

        # if it's a section marker
        if 'section' in item:
            # put previous section elements in dict
            elements[cur_section] = cur_items
            # start a new section
            cur_section = item
            cur_items = []

        # else append element to items in existing section
        else:
            cur_items.append(item) 

    # do this for the last section also
    elements[cur_section] = cur_items

    return elements


def page_sections(id_list, label):
    """Input: 
         [all page_title_section_id in a set]
         label for all sections in set (0/1)
       Output:
         {page: [{"section_id": section_id, "label": label}]}
    """

    pages = {}

    for element in id_list:

        page_title = element.split("_")[0]
        section_id = "_".join(element.split("_")[1:])

        if page_title in pages:
            pages[page_title].append({"section_id": section_id, 
                                      "label": label})
        else:
            pages[page_title] = [{"section_id": section_id,
                                  "label": label}]

    return pages 


def mk_sentences(db, claim, page_title, relevant_sections, non_relevant_sections):
    """Input: page_title, [relevant_section_ids], [non_relevant_section_ids]
       Output: 
        - page_index: {"page_id_section_id|sentence_id": sentence} for all sentences
        in that section
        - page_labels: {"page_id_section_id|sentence_id": label} for all sentences
        in that section
       Label is 0 for not relevant or 1 for relevant
    """

    page_index = {}
    page_labels = {}

    page_json = db.get_doc_json(page_title)
    wiki_page = WikiPage(page_title, page_json)

    # get {section: [elements]} dict for page
    page_elements = page_json["order"]
    elements = section_off_page(page_elements)

    # now get the required return dicts
    for section_id, section_items in elements.items():

        # get label of section
        if section_id in relevant_sections:
            section_label = 1
        elif section_id in non_relevant_sections:
            section_label = 0
        else:
            # if section not in either then don't consider it at all
            continue

        # iterate through all items in that section
        for item_id in section_items:
            full_item_id = claim + "|" + page_title + "_" + section_id + "|" + item_id
            item = str(wiki_page.get_element_by_id(item_id))
            
            # putting item in the required return dicts
            page_index[full_item_id] = item
            page_labels[full_item_id] = section_label
    
    return page_index, page_labels, elements


def mk_sentence_docs(db, datapoint):
    """Steps 1 and 1.5 for a single claim and json"""

    # claim
    claim = datapoint["claim"]
    # list of evidence sections
    evidence = datapoint["evidence"]
    # list of not-evidence sections
    not_evidence = datapoint["not_evidence"]

    # {page: [{section_id, label}]} grouping
    pages_evidence = page_sections(evidence, 1)
    pages_not_evidence = page_sections(not_evidence, 0)

    # put both into one dict
    pages = pages_evidence
    for page, sections in pages_not_evidence.items():
        if page in pages:
            pages[page] += sections
        else:
            pages[page] = sections

    # okay now do steps 1, 1.5 for every page in dict
    # and accumulate them
    index = {}    # step 1
    labels = {}    # step 1.5
    all_sections = {}   # section: [items] dict

    # for each page in pages
    for page, sections in pages.items():

        relevant = list(set([s["section_id"] for s in sections if s["label"] == 1]))
        not_relevant = list(set([s["section_id"] for s in sections if s["label"] == 0]))

        page_index, page_labels, sections_items = mk_sentences(db, claim, page, relevant, not_relevant)
        index.update(page_index)
        labels.update(page_labels)
        all_sections[page] = sections_items
    
    return index, labels, all_sections


def tfidf_retrieve_n(db, n, doc_ids, cosine_sims):
    """Input:
            n :: int
            [doc_ids]
            [cosine_similarities] corresponding to doc_ids
       Output: 
            [top n doc_ids for section-claim pair]}
    """
    
    # get page
    page_title = doc_ids[0].split("|")[1].split("_")[0]

    # retrieve sentence_ids from doc_ids
    sentence_ids = [id.split("|")[2] for id in doc_ids]
    
    top_n_idx = np.argsort(cosine_sims)[:5]
    top_n_ids = [sentence_ids[i] for i in top_n_idx]
    
    return top_n_ids


def sentences_to_sequences(db, section_items, sentences):
    """Input: 
        wikipedia DB object
        [items in this particular section]
        [top_n_sentence_ids]
      Output: For all sentences in one particular wiki page's section, return the following:
        [sentence before, sentence(s), sentence after]
        UNLESS element is not a sentence, in which case return
        only [element]"""

    # get sentential items in section only
    section_sents = [item_id for item_id in section_items if 'sentence' in item_id]

    # sentential elements in `sentences`, sorted
    sentence_ids = sorted([doc_id for doc_id in sentences if 'sentence' in doc_id], key=lambda x: int(x.split("_")[1]))
    # non-sentential elements in `sentences`
    other_elements = [doc_id for doc_id in sentences if 'sentence' not in doc_id]

    seqs = []
    cur_seq = []
    for sentence_id in sentence_ids:

        # get index of sentence in section items
        idx = section_sents.index(sentence_id)
    
    # ======== IF-ELSE TREE FOR CREATING SEQUENCES =======

        # SPECIAL CASE: if last element in list, 
        # 1. append sentence_id to seq
        # 2. END: append [prev_sentence[seq[0]], seq] to seqs
        # this can apply for element 0 in singleton lists
        if idx == len(section_sents) - 1:
            cur_seq.append(sentence_id)

            # preceding element is element preceding first element of seq, UNLESS first element of seq is at index 0
            # no succeeding element for this case
            prev_idx = section_sents.index(cur_seq[0]) - 1
            if prev_idx >= 0:
                prev = section_sents[prev_idx]
                cur_seq = [prev] + cur_seq

            # append current sequence to sequence list
            seqs.append(cur_seq)
            cur_seq = []

        # SPECIAL CASE: if index is 0, AND LIST IS NOT SINGLETON, append sentence to seq
        elif idx == 0:
            cur_seq.append(sentence_id)

        # NORMAL CASE: if seq is empty, then add current element
        # to seq
        elif cur_seq == []:
            cur_seq.append(sentence_id)

        # NORMAL CASE: if seq is not empty and current element immediately follows last element in seq
        # then add current element to seq
        elif idx == (section_sents.index(cur_seq[-1]) + 1):
            cur_seq.append(sentence_id)

        # NORMAL CASE: if seq is not empty and current element does not immediately follow last element in seq
        # then append [prev, seq, next] to sequences
        # and append this element to new seq
        elif idx > (section_sents.index(cur_seq[-1]) + 1):

            # `prev` is element preceding first element of seq, UNLESS first element of seq is at index 0
            prev_idx = section_sents.index(cur_seq[0]) - 1
            if prev_idx >= 0:
                prev = section_sents[prev_idx]
                cur_seq = [prev] + cur_seq
            # `next` is element succeeding last element of seq UNLESS last element of seq is last element of items list
            next_idx = section_sents.index(cur_seq[-1]) + 1
            if next_idx < len(section_sents):
                nxt = section_sents[next_idx]
                cur_seq = cur_seq + [nxt]
            
            seqs.append(cur_seq)
            cur_seq = [sentence_id]

        # if any other case occurs, there's a bug in my code
        else:
            print("---------- Bug starts here ------------")
            print(cur_seq)
            print(seqs)
            print(sentence_id)
            raise Exception("There's a bug in my code")

    # ======= END IF-ELSE TREE FOR CREATING SEQUENCES =======

    # append non-sentential elements to seqs also
    for el in other_elements:
        seqs += [[el]]

    # print(section_id)
    # print("-------------------------------")
    # for seq in seqs:
    #    print(seq)
    #    print("-------------------------------")
    # print("===================================")
    return seqs


# ====================== Main ==============================


if __name__ == "__main__":

    db = FeverousDB(args.db_path)

    # open file to write to
    # writefile = open(args.write, "w+")

    # read in all data
    with open(args.train) as sectionsf:
        data = [json.loads(sectionsf.readline()) for i in range(args.lines)]

    # print("Step 1: Making {claim|page_id_section_id|sentence_id: sentence} dict")
    # sentence_datapoints = {}
    # sentence_labels = {}
    # all_section_items = {}
    # for point in tqdm(data):
    #     index, labels, page_section_items = mk_sentence_docs(db, point)
    #     sentence_datapoints.update(index)
    #     sentence_labels.update(labels)
    #     all_section_items.update(page_section_items)
        
    # # step 1.5: condense labels down to section_labels
    # # i.e: {claim|page_id_section_id: label}
    # # please test print to understand what is going on here
    # section_labels = {}
    # for i, label in sentence_labels.items():
    #     i_ = "|".join(i.split("|")[:2])
    #     section_labels[i_] = label

    # # step 2.1: make [id], [sentence] lists for dict in step 1
    # items = sentence_datapoints.items()
    # ids = [pair[0] for pair in items]
    # sentences = [pair[1] for pair in items]
    
    # # step 2.2: Preprocess all items in [sentence] to prepare for TF-IDF vectorization
    # print("Step 2: preprocess {N} docs with NLTK".format(N=len(sentences)))
    # preproc_sentences = []
    # for sentence in tqdm(sentences):
    #     preproc_sentences.append(preprocess(sentence))

    # # UNCOMMENT SECTION BELOW IF PREPROCESSING WITH SPACY
    # # print("Step 2: preprocess {N} docs with Spacy".format(N=len(sentences)))
    # # docs = nlp.pipe(sentences, n_process=args.processes)
    # # print("Documents preprocessing...")

    # # step 2.3: and make a new list corresponding to [ids]:
    # # [preproc_sentences]
    # # preproc_sentences = []
    # # docs_iterator = tqdm(docs)
    # # for doc in docs_iterator:
    # #    preproc_sentences.append(" ".join([token.lemma_ for token in doc]))

    # # step 3: tf-idf vectorize list from step 3
    # print("Step 3: performing tf-idf vectorization over sentences")
    # vectorizer = TfidfVectorizer()
    # tfidf_matrix = vectorizer.fit_transform(preproc_sentences) 

    # save_npz("tfidf_matrix.npz", tfidf_matrix)
    # with open("ids.json", "w+") as wf:
    #     json.dump(ids, wf)
    # with open("labels.json", "w+") as lf:
    #     json.dump(section_labels, lf)
    # claims = [preprocess(d['claim']) for d in data]
    # claims_vectors = vectorizer.transform(claims)
    # save_npz("claims_vectors.npz", claims_vectors)


    # =============== PART 2 OF FILE ===================

    claims = [d['claim'] for d in data]
    claim_vecs = load_npz("claims_vectors.npz")
    tfidf_matrix = load_npz("tfidf_matrix.npz")
    ids = json.load(open("ids.json"))
    labels = json.load(open("labels.json"))

    # step 4: get {claim|section_id: [indexes]} set out of `ids`
    print("Step 4: get {claim|section_id: [indexes]} set out of ids")

    claim_section_idx = {}
    for idx, id in enumerate(ids):
        clm_sec = "|".join(id.split("|")[:2])
        sent_id = id.split("|")[-1]
        
        if clm_sec in claim_section_idx:
            claim_section_idx[clm_sec].append(idx)
        else:
            claim_section_idx[clm_sec] = [idx]

    # step 5: get cosine_similarity across all claims and sentences
    print("Step 5: for claim, section in claims_sections, get top N sentences")
    for c_s, idx in tqdm(claim_section_idx.items()):

        claim = c_s.split("|")[0]
        claim_idx = claims.index(claim)
        claim_vec = claim_vecs[claim_idx] 

        page_title = section = c_s.split("|")[1].split("_")[0]
        section_id = "_".join(c_s.split("|")[1].split("_")[1:])

        # get sentence_ids also corresponding to vectors
        doc_ids = [ids[i] for i in idx]

        # # 5.2: calculate cosine_sim(claim_vector, tf_idf vectors)
        cosine_sims = cosine_similarity(tfidf_matrix, claim_vec).flatten()
"""
        # 5.3: get top N sentence IDs
        top_n_sent_ids = tfidf_retrieve_n(db, args.n, doc_ids, cosine_sims)

        # Step 6: get top `n` sequences as IDS
        top_n_seqs = sentences_to_sequences(db, all_section_items[page_title][section_id], top_n_sent_ids)

        # convert everything in sequences to actual text
        page_json = db.get_doc_json(page_title)
        wiki_page = WikiPage(page_title, page_json)

        top_n_text = []
        for ls in top_n_seqs:
            ls_text = []
            for id in ls:
                ls_text.append(str(wiki_page.get_element_by_id(id)))
            top_n_text.append(ls_text)

        # now write output to file in above format
        write_dict = {"claim": claim,
                      "data": 
                        {"section_id": page_title + "|" + section_id,
                         "sequences": top_n_text,
                         "label": section_labels[c_s]
                        }
                      }

        writefile.write(json.dumps(write_dict) + "\n")
"""