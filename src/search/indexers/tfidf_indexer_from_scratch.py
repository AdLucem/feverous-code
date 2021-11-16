import json
import jsonlines
import math
import spacy
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage

FILENAME = "/home/kw/Data/FEVEROUS/dev_pages/dev.pages.p70.jsonl"

db = FeverousDB("/home/kw/Data/FEVEROUS/feverous_wikiv1.db")

nlp = spacy.load("en_core_web_sm", exclude=["parser"])
stopwords = nlp.Defaults.stop_words

porter_stemmer = PorterStemmer()


# =================== FUNCTIONS =====================


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


def make_inverted_index(docs, num_processes):
    """Input:
        docs: {doc_id: [sentences]}
       Output:
        index := {token: {doc_id: num_instances}}
    """

    index = {}
    page_docs = {}

    # iterate over documents
    for doc_id in docs:
        sentences = docs[doc_id]
        spacy_docs = nlp.pipe(sentences, n_process=num_processes)

        # append to documents list
        page_docs[doc_id] = []

        # for each sentence
        for spacy_doc in spacy_docs:

            # tokens, stemmed, without stopwords
            # TODO: do some more cleaning here
            tokens_preproc = [porter_stemmer.stem(tok.text) for tok in spacy_doc if nlp.vocab[tok.text].is_stop == False]
            # put tokens in doc
            page_docs[doc_id].append(tokens_preproc)

            # for each non-stopword token in sentence
            for token in tokens_preproc:

                # if token not in inverted index
                if token not in index:
                    index[token] = {}
                    index[token][doc_id] = 1

                # elif token in index but not under this doc_id
                elif doc_id not in index[token]:
                    index[token][doc_id] = 1

                # elif token, doc_id already in index
                else:
                    index[token][doc_id] += 1

    # test-print inverted index
    # for t in index:
    #    print(t)
    #    print("------------------------------------")
    #    for docid, num in index[t].items():
    #        print("%20s %5d" % (docid, num))
    #    print("------------------------------------")
    return page_docs, index


def idf(term, inverted_index, docs):
    """idf(term) = log(num. documents in corpus/count of term in corpus) + 1"""

    # N := num_documents_in_corpus
    N = len(list(docs.keys()))

    inv_index_term = inverted_index[term]
    df = sum([inv_index_term[doc_id] for doc_id in inv_index_term])

    idf_score = math.log(N / (df + 1))

    return idf_score


def tf(term, doc_id, inverted_index, docs):
    """tf(term, doc) = count of term in doc/total num. words in doc
    """

    # count of term in doc
    count = inverted_index[term][doc_id]

    # total num of words in doc
    n_words = sum([len(sent) for sent in docs[doc_id]])

    tf_score = count / n_words

    return tf_score


def make_tf_idf(inverted_index, docs):
    """Convert from inverted index to TF-IDF index
        Input:
            inverted_index := {token: {doc_id: num_occurences}}
        Output:
            tfidf := {token: {doc_id: tf_idf_score}}
    """

    tfidf = {}
    for term in inverted_index:

        tfidf[term] = {}

        idf_score = idf(term, inverted_index, docs)

        doc_ids = inverted_index[term]
        for doc_id in doc_ids:
            tf_score = tf(term, doc_id, inverted_index, docs)

            tfidf[term][doc_id] = tf_score * idf_score

    # test print tf-idf
    # for term in tfidf:
    #    print(term)
    #    print("------------------------------------")
    #    for doc, score in tfidf[term].items():
    #        print("%20s   %8f" % (doc, score))
    #    print("====================================")


def index_for_sample(sample_pages, num_processes):
    """Make inverted index for all pages in sample"""

    # overall inverted index
    inv_index = {}
    # all {doc_id: [sentences]}
    docs = {}

    for page in sample_pages:
        # get each section of page as its own doc
        section_docs = get_page_sections_as_docs(db, page)

        # now get tokens and make inverted index for these docs
        page_docs, page_inv_index = \
            make_inverted_index(section_docs, num_processes)
        docs.update(page_docs)

        # integrate the page-specific inv.index with the overall index
        # for each token in page
        for token in page_inv_index:

            # if token already in overall inv.index
            if token in inv_index:
                # combine section:num dictionary with overall one
                inv_index[token].update(page_inv_index[token])

            # else if token new to inverted index
            else:
                # make initial inv.index dict of token the same
                # as the page-specific inv. index of token
                inv_index[token] = page_inv_index[token]

    # test-print docs
    # for doc in docs:
    #    print(doc)
    #    print("===================================")
    #    for sent in docs[doc]:
    #        print(sent)
    #    print("-----------------------------------")
    # test-print overall inverted index
    # for tok in inv_index:
    #    print(tok)
    #    print("------------------------------------")
    #    for docid, num in inv_index[tok].items():
    #        print("%20s %5d" % (docid, num))
    #    print("====================================")
    return docs, inv_index


def make_into_docs(sample_pages, num_processes):
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


def search(claim, tfidf_matrix, n):

    # Vectorize the claim to the same length as documents
    claim_vec = vectorizer.transform([claim])
    # cosine similarity between claim_vec and all the documents
    cosine_sims = cosine_similarity(tfidf_matrix, claim_vec).flatten()
    # Sort the similar documents from the most similar
    # to less similar and return the indices
    most_similar_doc_indices = np.argsort(cosine_sims, axis=0)[:n]
    return most_similar_doc_indices


# ===================== MAIN =========================
if __name__ == "__main__":

    retrieved_sections = []
    N = 10

    with open(FILENAME) as f:
        data = [json.loads(line) for line in f.readlines()]
    
    for point in data:
        
        claim_id = point['id']
        claim = point['claim']
        page_tuples = point["predicted_pages"]
            pages = [page[0] for page in page_tuples]

            docs = make_into_docs(pages, 4)
            docs_tuples = docs.items()
            doc_ids = [item[0] for item in docs_tuples]
            doc_contents = [item[1] for item in docs_tuples]

            vectorizer = TfidfVectorizer()
            tfidf_matrix = \
                vectorizer.fit_transform(doc_contents)
            # print(vectorizer.get_feature_names())

            most_similar_doc_indices = search(claim, tfidf_matrix, N)
            predicted_sections = [doc_ids[idx] for idx in most_similar_doc_indices]

            retrieved = {"claim_id": claim_id,
                         "claim": claim,
                         "predicted_sections": predicted_sections}
            retrieved_sections.append(retrieved)

    with jsonlines.open("top_" + str(N) + "_sections.json", "w+") as writer:
        writer.write_all(retrieved_sections)
