import json
import joblib
import argparse
import time
from tqdm import tqdm
from os.path import join
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage


parser = argparse.ArgumentParser()

parser.add_argument('--top_pages', type=str, help="Path to file containing top N pages for each claim")
parser.add_argument('--save_dir', type=str, help="Save TF-IDF vectors to directory")
parser.add_argument('--db_path', type=str, help="Path to feverous_wikiv1.db")

args = parser.parse_args()

# initialize feverous database
db = FeverousDB(args.db_path)
# initialize stemmer
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


def make_into_docs(sample_pages):
    """Make section-based docs for all pages in sample"""

    # all {doc_id: doc_text}
    docs = {}

    print("Getting all docs from pages")
    for page in tqdm(sample_pages):
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


def mk_inverted_index(docs):
    """Make inverted index vector matrix for all docs in given 
       {doc_id: doc_text} mapping
       Return: [doc_ids], [doc vectors]"""

    docs_tuples = docs.items()

    doc_ids = [item[0] for item in docs_tuples]
    doc_contents = [item[1] for item in docs_tuples]

    print("Making inverted index from docs")
    vectorizer = TfidfVectorizer(stop_words="english")

    t0 = time.time()
    tfidf_matrix = vectorizer.fit_transform(doc_contents)
    t1 = time.time()

    print(len(doc_ids), "docs converted to TF-IDF vectors in", 
          str(t1 - t0), "time")
    print("Made TF-IDF matrix of shape (num_docs, vocab):", tfidf_matrix.shape)

    return doc_ids, tfidf_matrix


def run():
    """Make section-level inverted index for all sections
       in all pages related to claims in filename
       Actions:
        saves doc_ids list, inverted_index matrix that can be used for 
        section retrieval
    """

    with open(args.top_pages) as f:
        data = [json.loads(line) for line in f.readlines()]

    all_pages = []
    for point in data:
        page_tuples = point["predicted_pages"]
        pages = [page[0] for page in page_tuples]
        all_pages += pages

    docs = make_into_docs(all_pages)
    doc_ids, doc_vectors = mk_inverted_index(docs)

    # save doc IDS as json file
    save_doc_ids = join(args.save_dir, "doc_ids.json")
    with open(save_doc_ids, "w+") as f:
        json.dump(doc_ids, f)

    # save tf-idf matrix as joblib file
    save_doc_vectors = join(args.save_dir, "doc_vectors.joblib")
    joblib.dump(doc_vectors, save_doc_vectors)


def load_joblib(filename):
    """Load an existing vectors file from joblib"""

    vectors = joblib.load(filename)
    print("Confirmed that vectors file loads: shape:", vectors.shape)


# ===================== MAIN =========================
if __name__ == "__main__":

    run()

    save_doc_vectors = join(args.save_dir, "doc_vectors.json")
    load_joblib(save_doc_vectors)
"""
    retrieved_sections = []

    with open(FILENAME) as f:

        maybe_line = f.readline()

        # while maybe_line != "":
        for i in range(4):

            line = json.loads(maybe_line)
            maybe_line = f.readline()

            claim_id = line['id']
            claim = line['claim']
            page_tuples = line["predicted_pages"]
            pages = [page[0] for page in page_tuples]

            docs = make_into_docs(pages)
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

    with jsonlines.open("data/top_" + str(N) + "_sections.jsonl", "w") as writer:
        writer.write_all(retrieved_sections)
"""