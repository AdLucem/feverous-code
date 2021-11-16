import json
from tqdm import tqdm
import pandas as pd
from os import mkdir
from os.path import join, isdir

# Used to create the dense document vectors.
import torch
from sentence_transformers import SentenceTransformer

# wikipedia page-processing utility functions
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


def make_into_docs(db, sample_pages):
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


def get_device_details():
    """Get details of current devices that torch + transformers
    is running on. Return True if GPU available"""

    print("Device details")
    # Check if CUDA is available ans switch to GPU
    is_gpu = False
    if torch.cuda.is_available():
        is_gpu = True
        print("GPU")
    if is_gpu:
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
        print("========================================================")
    else:
        print("CPU")

    return is_gpu


def mk_vector_index(docs):
    """Make inverted index vector matrix for all docs in given 
       {doc_id: doc_text} mapping
       Return: [doc_ids], [doc vectors]"""

    # Instantiate the sentence-level DistilBERT
    print("Instantiate the sentence-level")
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    is_gpu_there = get_device_details()
    if is_gpu_there:
        model = model.to(torch.device("cuda"))

    # get all the doc texts as list
    docs_items = list(docs.items())

    # get all the doc ids
    doc_ids = [x[0] for x in docs_items]

    # convert to list of lists
    data_as_ndarray = [[x[0], x[1]] for x in docs_items]
    df = pd.DataFrame(data_as_ndarray, columns=["id", "text"])
   
    # convert doc texts to vectors
    embeddings = model.encode(df.text.to_list(), show_progress_bar=True)
    print("Encoded {LENGTH} documents as vectors of dim {DIM}".format(LENGTH=len(data_as_ndarray), DIM=embeddings.shape[1]))

    return doc_ids, embeddings


def run_vector_indexer(read_path, index_dir, db_path):
    """Main function for vector indexing
        read_path: path to top 20 pages per claim
        write_path: directory to write index to
        db_path: path to feverous_wikiv1.db
        split: whether to split claims before indexing or not"""

    # initialize feverous database
    print("Initializing feverous database...")
    db = FeverousDB(db_path)

    print("Reading in data...")
    with open(read_path) as f:
        data = [json.loads(line) for line in f.readlines()]

    all_pages = []
    for point in data:
        page_tuples = point["predicted_pages"]
        pages = [page[0] for page in page_tuples]
        all_pages += pages

    print("Make docs out of all pages")
    docs = make_into_docs(db, all_pages)
    doc_ids, doc_vectors = mk_vector_index(docs)

    write_vectors = join(index_dir, "doc_vectors.pt")
    write_ids = join(index_dir, "doc_ids.json")

    if not isdir(index_dir):
        mkdir(index_dir)

    with open(write_ids, "w+") as f:
        json.dump(doc_ids, f)
    print("Saved document IDs- as JSON- in", write_ids)

    with open(write_vectors, "wb+") as f:
        torch.save(doc_vectors, f)
    print("Saved document vectors- as pytorch tensors- in", write_vectors)
