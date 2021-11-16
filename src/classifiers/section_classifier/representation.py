"""Representation of each section/claim/label triplet as a sequence that can be fed to RoBERTa"""

# step 1: convert all the data into a {page_id_section_id|sentence_id: sentence} sequence. Same for table cells.
# step 1.5: make a separate dict with 
#                {page_id_section_id: label}
# step 2: given the above (doc: content) pairs, make a TF-IDF index
# step 3: for section in claim_sections:
#             top_n_sentences = tfidf_retrieve_n(claim, sentences_in_section)
# Step 4: out of top_n_sentences, make:
#    top_n_sequences = [(sentence_before, sentence, sentence_after) for sentence in top_n_sentences]
# Output: {claim_id, 
#          claim, 
#          [{sectionid, [sequences], label}]}

import argparse
import json
from tqdm import tqdm

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


parser = argparse.ArgumentParser()

parser.add_argument('--train', type=str, help="Path to training data file")
parser.add_argument('--write', type=str, help="Path to file to write to")
parser.add_argument('--db_path', type=str, help="Path to feverous_wikiv1.db")
parser.add_argument('--n', type=int, help="Number of sentences per section")
parser.add_argument('--lines', type=int, help="Number of lines in training data file")
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


def mk_sentences(db, page_title, relevant_sections, non_relevant_sections):
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
            full_item_id = page_title + "_" + section_id + "|" + item_id
            item = str(wiki_page.get_element_by_id(item_id))
            
            # putting item in the required return dicts
            page_index[full_item_id] = item
            page_labels[full_item_id] = section_label
    
    return page_index, page_labels, elements


def mk_training_data(db, datapoint):
    """Steps 1 and 1.5 for a single json in file"""

    # list of evidence sections
    evidence = line_data["evidence"]
    # list of not-evidence sections
    not_evidence = line_data["not_evidence"]

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

        page_index, page_labels, sections_items = mk_sentences(db, page, relevant, not_relevant)
        index.update(page_index)
        labels.update(page_labels)
        all_sections[page] = sections_items
    
    return index, labels, all_sections


def mk_tfidf_vectorizer(index):
    """Input:
        {doc_id: doc} dict where each doc is a single sentence
       Output:
        [doc_ids]
        tfidf_vectorizer
    """

    doc_items = index.items()
    doc_ids = [item[0] for item in doc_items]
    docs = [item[1] for item in doc_items]

    # preprocess docs
    docs_preproc = [preprocess(doc) for doc in docs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs_preproc)

    return doc_ids, tfidf_matrix, vectorizer


def tfidf_retrieve_n(n, doc_ids, cosine_sims):
    """Input:
            n :: int
            [doc_ids]
            [cosine_similarities] corresponding to doc_ids
       Output: 
            {section_id: [top n doc_ids for section]}
    """
    
    # retrieve section_ids from doc_ids
    section_ids = list(set([id.split("|")[0] for id in doc_ids]))
    
    # dict for {section_id: [top n]}
    top_n_overall = {}

    # so over all section ids
    for sec_id in section_ids:

        # get doc_index, doc_id with given sec_ids
        index_id_pairs = [index_id_pair for index_id_pair in enumerate(doc_ids) if (sec_id == index_id_pair[1].split("|")[0])]

        # get cosine similarities for all of these docs only
        # actually do a loop instead of a list comprehension here
        # for readability reasons
        sims = []  # [(doc_id, cosine_sim)]
        for index, doc_id in index_id_pairs:
            sims.append((doc_id, cosine_sims[index]))

        # get the top N OR FEWER similar sentences in that section
        sorted_by_sim = sorted(sims, key=lambda item: item[1], reverse=True)
        top_n = [sort_pairs[0] for sort_pairs in sorted_by_sim[:n]]

        top_n_overall[sec_id] = top_n

    return top_n_overall


def sentences_to_sequences(db, pages_sections_items, sentences):
    """Input: 
        wikipedia DB object
        {page: {section_id: [items]}}
        [sentences for a particular section in particular page]
      Output: For all sentences in one particular wiki page, return the following:
        [sentence before, sentence(s), sentence after]
        UNLESS element is not a sentence, in which case return
        only [element]"""

    section_id = sentences[0].split("|")[0]
    page_title = section_id.split("_")[0]
    page_json = db.get_doc_json(page_title)
    wiki_page = WikiPage(page_title, page_json)
    section_id_only = "_".join(section_id.split("_")[1:])

    # sentences only in section
    section_items = pages_sections_items[page_title][section_id_only]
    section_sents = [item_id for item_id in section_items if 'sentence' in item_id]

    # sentential elements in doc, sorted
    sentence_ids = sorted([doc_id.split("|")[1] for doc_id in sentences if 'sentence' in doc_id], key=lambda x: int(x.split("_")[1]))
    # non-sentential elements in doc
    other_elements = [doc_id.split("|")[1] for doc_id in sentences if 'sentence' not in doc_id]
    
    # print(sentence_ids)
    # print(section_sents)
    # print("+++++++++++++++++++++++++++++++++")

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
    top_n_text = []
    for ls in seqs:
        ls_text = []
        for id in ls:
            ls_text.append(str(wiki_page.get_element_by_id(id)))
        top_n_text.append(ls_text)

    return top_n_text


# ====================== Main ==============================


if __name__ == "__main__":

    db = FeverousDB(args.db_path)

    # open file to write to
    writefile = open(args.write, "w+")

    """NOTE: COMMENT OUT THIS BIT IF RUNNING ON ADA"""
    # iterate through lines in training data file
    # with open(args.train) as sectionsf:
    #    maybe_line = sectionsf.readline()
    #    for i in tqdm(range(args.lines)):
    #    line_data = json.loads(maybe_line)
    #    maybe_line = sectionsf.readline()

    """NOTE: COMMENT OUT THIS BIT IF RUNNUNG ON LOCAL PC"""
    times = []
    with open(args.train) as sectionsf:
        lines_data = [json.loads(maybe_line) for maybe_line in sectionsf.readlines()]
        for i in tqdm(range(args.lines)):
            line_data = lines_data[i]    
            
            # step 1
            index, sentence_wise_labels, section_items = mk_training_data(db, line_data)

            # step 1.5: condense labels down to section_labels
            # i.e: {page_id_section_id: label}
            # please test print to understand what is going on here
            labels = {}
            for i, label in sentence_wise_labels.items():
                i_ = i.split("|")[0]
                labels[i_] = label

            # step 2
            doc_ids, tfidf_matrix, vectorizer = mk_tfidf_vectorizer(index)

            # step 3
            claim = line_data['claim']
            claim_preproc = preprocess(claim)
            # Vectorize the claim to the same length as documents
            claim_vec = vectorizer.transform([claim])
            # cosine similarity between claim_vec and all the documents
            cosine_sims = cosine_similarity(tfidf_matrix, claim_vec).flatten()

            # step 3 - retrieve top N sections for each section
            top_n_sentences = tfidf_retrieve_n(args.n, doc_ids, cosine_sims)

            # step 4
            # NOTE: sequences may be empty list
            sections_data = []
            for sec_id, sentences in top_n_sentences.items():

                top_n_sequences = sentences_to_sequences(db, section_items, sentences)
                sec_label = labels[sec_id]

                sections_data.append({"section_id": sec_id,
                                      "sequences": top_n_sequences,
                                      "label": sec_label})

            # now make a json of the form:
            # {claim_id, 
            # claim, 
            # [{sectionid, [sequences], label}]}
            write_line = {"id": line_data["id"],
                          "claim": line_data["claim"],
                          "sections_data": sections_data}  
            # aaaaaaand write this to file
            writefile.write(json.dumps(write_line) + "\n")

    writefile.close() 
