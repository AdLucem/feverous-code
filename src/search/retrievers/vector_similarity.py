import json
from os.path import join
from tqdm import tqdm

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage
    

class VectorSimilarity:

    def __init__(self):

        self.model = self.init_model()

    def init_model(self):
        """Instantiate the sentence-level DistilBERT"""

        print("Instantiate the sentence-level model")
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        if torch.cuda.is_available():
            model = model.to(torch.device("cuda"))

        return model

    def vectorize_all(self, lsstr):
        """Vectorize all strings in a list, as a tensor of shape 
           (len_list, num_features)"""

        print("Encoding {num} strings".format(num=len(lsstr)))
        vectors = self.model.encode(lsstr, show_progress_bar=True)

        return vectors

    def retrieve(self, query, search_space, N):
        """Given a query and a search space, retrieves N most relevant docs
           and returns [relevant_doc_ids]"""

        # get cosine similarity between query vector and all document vectors
        cosim = cosine_similarity(query.vector, search_space.doc_vectors)

        # get indices of top N similarity scores
        top_sims = np.argsort(np.reshape(cosim, (cosim.shape[1]))).tolist()[-N:]

        # get section_ids corresponding to indices in top_sims
        top_doc_ids = [search_space.doc_ids[i] for i in top_sims]

        return top_doc_ids


class DataPoint:

    def __init__(self, datapoint_in_json, db):

        self.claim = datapoint_in_json["claim"]
        self.split_claim = datapoint_in_json["split_claim"]
        self.docs = self.get_section_docs(datapoint_in_json, db)

    def get_section_docs(self, datapoint_in_json, db):
        """get subsection docs- {doc_id: doc_text} for this claim only"""

        pages = [x[0] for x in datapoint_in_json["predicted_pages"]]
        section_docs = make_into_docs(db, pages, 4)
        return section_docs

    def populate_subclaims(self, queries):
        """Given a list of [Query] objects, take all the queries that are
           subclaims of this claim"""

        self.subclaims = [q for q in queries if q.parent_datapoint == self]

    def print_subclaims(self):

        print(self.claim)
        print("-----------------------")
        for subclaim in self.subclaims:
            print(subclaim.query)
        print("=======================")

    def unionize(self, N, method):
        """Take top-N-union-reranking-and-retrieval of relevant-docs in all subclaims using a given unionizing method and given value of N"""

        if method == "simple":
            unionizer = Union(self)
            union_set = unionizer.simple_union(N)

        self.relevant_doc_ids = union_set

    def __repr__(self):

        doc_ids = list(self.docs.keys())

        s = self.claim + "\n"
        s += str(self.split_claim) + "\n"
        s += "--------------------------------\n"

        for i in range(0, len(doc_ids), 3):
            if (i + 2) < len(doc_ids):
                s += "%25s $ %25s $ %25s\n" % (doc_ids[i], 
                                               doc_ids[i + 1], 
                                               doc_ids[i + 2])
            else:
                for doc_id in doc_ids[i:]:
                    s += "%25s $ " % (doc_id)
                s += "\n"
        s += "=============================================="
                
        return s 


class Query:
    """A single query- with index and associated main claim"""

    def __init__(self, index, query, parent_claim):

        self.index = index
        self.query = query
        self.parent_datapoint = parent_claim

        self.vector = []
        self.relevant_doc_ids = []

    def __repr__(self):

        s = "{IDX}. {QUERY}\n".format(IDX=str(self.index), QUERY=self.query)
        if self.vector != []:
            s += "Vector of size {SHAPE}\n".format(SHAPE=self.vector.shape)
        s += "--------------------------------\n"
        s += self.parent_datapoint.claim + "\n"
        s += "+++++++++++++++++++++++++++++++++++++++++++\n"
        return s

    def retrieve_and_populate(self, retriever, search_space, N):
        """Given a search space, populate top N most relevant doc_ids for the query
           and populate self.relevant_doc_ids"""

        self.relevant_doc_ids = retriever.retrieve(self, search_space, N)

    def print_relevant_docids(self):

        print("{IDX}. {QUERY}\n".format(IDX=str(self.index), QUERY=self.query))
        print("------------------------------------------------")
        for doc_id in self.relevant_doc_ids:
            print(doc_id)
        print("================================================")

    @staticmethod
    def from_list(datapoints):
        """From a list: [DataPoint], make a list [Query]"""

        split_claims = [point.split_claim for point in datapoints]
       
        # make list of Queries, indexed by main claim and claim ID
        queries = [] 
        for i, split_claim in enumerate(split_claims):
            for subclaim in split_claim:
                query = Query(i, subclaim, datapoints[i])
                queries.append(query)

        return queries

    @staticmethod
    def populate_vectors(retriever, queries):
        """STATE CHANGING FUNCTION: Vectorize all queries in a list [Query], and populate each individual query with vector""" 

        query_strings = [q.query for q in queries]
        query_vectors = retriever.vectorize_all(query_strings)

        for i, query_object in enumerate(queries):
            query_object.vector = np.reshape(query_vectors[i], 
                                             (1, query_vectors[i].shape[0]))      


class SearchSpace:
    """Initialize a search index- doc IDS and doc vectors- from a directory
       containing doc_ids.json and doc_vectors.pt"""

    def __init__(self, index_dir="", doc_ids=[], doc_vectors=[], from_file=True):
        """retrieve doc ids, vectors"""

        if from_file:
            doc_id_fname = join(index_dir, "doc_ids.json")
            doc_vecs_fname = join(index_dir, "doc_vectors.pt")

            self.doc_ids = json.load(open(doc_id_fname))
            self.doc_vectors = torch.load(doc_vecs_fname)

        else:
            self.doc_ids = doc_ids
            self.doc_vectors = doc_vectors

    def narrow_search_space(self, datapoint):
        """Narrow search space to ids and vectors of only the doc_ids of that datapoint""" 

        narrow_doc_ids = list(datapoint.docs.keys())

        # get indices of section doc vectors in the list
        narrow_doc_idxs = [self.doc_ids.index(id) for id in narrow_doc_ids]
        # get doc vectors of section docs for claim
        narrow_doc_vecs = [self.doc_vectors[idx].tolist() for idx in narrow_doc_idxs]
        # make numpy array out of this section docs subset
        narrow_vecs = np.array(narrow_doc_vecs)

        narrowed_search_space = SearchSpace(doc_ids=narrow_doc_ids,
                                            doc_vectors=narrow_vecs,
                                            from_file=False)
        return narrowed_search_space

    def __repr__(self):

        s = "Search space: {NUM_IDS} documents\n".format(NUM_IDS=len(self.doc_ids))
        s += "Using indexed document vectors of shape {SHAPE}\n".format(SHAPE=str(self.doc_vectors.shape)) 
        return s 


class Union:
    """Methods to get the union of all sections under all subclaims, for one single claim"""

    def __init__(self, datapoint):
        
        self.claim = datapoint.claim
        self.subclaims = datapoint.subclaims 

    def __repr__(self):
    
        s = self.claim + "\n"
        s += "----------------------------------\n"
        for subclaim in self.subclaims:
            s += subclaim.query + "\n"
        s += "==================================\n"
        return s

    def simple_union(self, N):
        """Take S = {union of all sections from all subclaims}. Do not rerank or retrieve top N"""

        S = []
        for subclaim in self.subclaims:
            S += subclaim.relevant_doc_ids

        return S 

    def ner_tfidf_union(self, N):
        """Take S = {union of all sections from all subclaims}, and use NER-tfidf-based-matching with the claim to rank sections within S. Take top N ranked sections"""

        print("TODO")


def get_page_sections_as_docs(db, title):
    """Get each section of the page as its own doc, in the format:
        {page title|section_id: text}"""

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


def run_split(db_path, datafile, indexdir, writefile, N, subN):
    
    print("Running on split claims...")

    # initialize feverous database
    db = FeverousDB(db_path)
    
    # Instantiate the retrieval machine
    retriever = VectorSimilarity()

    # load the entire search space
    whole_search_space = SearchSpace(index_dir=indexdir, from_file=True)
    print(whole_search_space)

    # retrieve top-20-pages data and make datapoints out of it
    with open(datafile) as f:
        datapoints = [DataPoint(json.loads(line), db) for line in f.readlines()]

    # sort and separate subclaims
    subclaim_queries = Query.from_list(datapoints)
    # make vectors for all subclaims
    Query.populate_vectors(retriever, subclaim_queries)
    
    # for subclaim in subclaims
    print("Retrieving top N docs for every subclaim...")
    for subclaim in tqdm(subclaim_queries):

        # narrow the search space to that particular claim
        sub_search_space = whole_search_space.narrow_search_space(subclaim.parent_datapoint)

        # get top <subN> most relevant docs for that particular subclaim
        subclaim.retrieve_and_populate(retriever, sub_search_space, subN)

    # get all subclaims associated with each query
    print("Take union of subclaims...")
    for point in tqdm(datapoints):
        point.populate_subclaims(subclaim_queries)

        # for each datapoint, take union of all subclaim-relevant-sections
        point.unionize(N, method="simple")

    # save
    save_data = [{"claim": point.claim, "predicted_sections": point.relevant_doc_ids} for point in datapoints]

    with open(writefile, "w+") as wf:
        for line in save_data:
            wf.write(json.dumps(line) + "\n")
    print("Saved {n} claim-predicted section JSONs in {fname}".format(n=len(datapoints), fname=writefile))


def run(db_path, datafile, indexdir, writefile, N):
    """FUNCTION NOT REFACTORED YET"""

    # initialize feverous database
    db = FeverousDB(db_path)
    
    # Instantiate the sentence-level DistilBERT
    print("Instantiate the sentence-level model")
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    # retrieve doc ids, vectors
    doc_id_fname = join(indexdir, "doc_ids.json")
    doc_vecs_fname = join(indexdir, "doc_vectors.pt")

    doc_ids = json.load(open(doc_id_fname))
    doc_vectors = torch.load(doc_vecs_fname)

    print("Using indexed document vectors of shape", str(doc_vectors.shape))

    # retrieve top-20-pages data
    with open(datafile) as f:
        data = [json.loads(line) for line in f.readlines()]

    # encode claims
    claims = [point["claim"] for point in data]
    print("Encoding {num} claims".format(num=len(claims)))
    claim_vectors = model.encode(claims, show_progress_bar=True)

    # take similarity between each claim and its sections
    i = 0
    claim_top_sections = []
    for point in tqdm(data):

        claim = point["claim"]
        claim_vector = np.reshape(claim_vectors[i], 
                                  (1, claim_vectors[i].shape[0]))
        pages = [x[0] for x in point["predicted_pages"]]

        section_docs = make_into_docs(db, pages, 4)
        section_doc_ids = list(section_docs.keys())

        try: 
            # get indices of section doc vectors in the index
            section_doc_idxs = [doc_ids.index(id) for id in section_doc_ids]

            # get doc vectors of section docs for claim
            section_doc_vecs = [doc_vectors[idx].tolist() for idx in section_doc_idxs]

            # make numpy array out of this section docs subset
            section_vecs = np.array(section_doc_vecs)

            # get indexes of top 20 most similar sections- by cosine sim
            # TEST PRINT print(claim_vector.shape, section_vecs.shape)
            cosim = cosine_similarity(claim_vector, section_vecs)
            top_sims = np.argsort(np.reshape(cosim, (cosim.shape[1]))).tolist()[-20:]

            # get top 20 sections by index
            top_sections = [section_doc_ids[i] for i in top_sims]

            claim_top_sections.append({"claim": claim, 
                                       "predicted_sections": top_sections})

        # if claim has no related pages, somehow
        except ValueError:
            print("Anomaly detected: claim has no related pages")

        finally:
            # increment index
            i += 1

    print(len(claim_top_sections))
    with open(writefile, "w+") as wf:
        json.dump(claim_top_sections, wf)
