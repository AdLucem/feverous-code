"""Create a BM-25 index for given data and retrieve top N pages, using BM-25 algorithm"""

from gensim import corpora

from gensim.summarization import bm25

from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation

import argparse

import jsonlines

import numpy as np

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--corpus_dir', type=str, default='data/corpus')
parser.add_argument('--dataset', type=str, default='data/dev.jsonl')
parser.add_argument('--k', type=int, default=10, help='top k wiki page to retrieve per claim')
parser.add_argument('--output_dir', type=str, default='predictions')
parser.add_argument('--remove_stopwords', type=str, default=False)
parser.add_argument('--section', type=str, default='all', help="options includes 'intro' and 'all'")

args = parser.parse_args()


def get_intro(doc):
    """get the intro section of a wiki page"""

    if 'section_0' in doc['order']:

        index = doc['order'].index('section_0') + 2

        intro = {k: doc[k] for k in list(doc)[:index]}

        elements_in_intro = get(intro)

    else:

        elements_in_intro = get(doc)

    return elements_in_intro


def get(doc):

    return " ".join([strip_punctuation(item) for item in [doc['title']] + get_sentences(doc) + get_tables(doc) + get_lists(doc)])


def get_lists(doc):

    l =[]

    for key, ele in doc.items():

        if key.startswith('list_'):

            l.extend([dic['value'] for dic in ele['list']])

    return l


def get_tables(doc):
    # get all the table values

    table = []

    for key, ele in doc.items():

        if key.startswith('table_'):

            table.extend([cell['value'] for row in ele['table'] for cell in row])

    return table


def get_sentences(doc):

    # sentences

    return [ele for key, ele in doc.items() if key.startswith('sentence_')]


if __name__ == '__main__':

    wiki_pages = []

    file_list = list(range(0, 1))   # test with only the first jsonl file

    # file_list = list(range(0, 528)) + list(range(530, 535)) + list(range(600, 611))       # use up the whole wiki dump

    print("The corpus has {} jsonl files".format(len(file_list)))

    for file_number in tqdm(file_list):

        wiki_pages.extend(list(jsonlines.open(args.corpus_dir + "/wiki_{:03}.jsonl".format(file_number))))

    print("The corpus has {} wiki pages".format(len(wiki_pages)))
"""
    if args.section == "all":

        # you can do preprocessing such as removing stopwords

        if args.remove_stopwords:

            texts = [remove_stopwords(get(page)).split() for page in wiki_pages]

        else:

            texts = [get(page).split() for page in wiki_pages]

    elif args.section == "intro":

        if args.remove_stopwords:

            texts = [remove_stopwords(get_intro(page)).split() for page in wiki_pages]

        else:

            texts = [get_intro(page).split() for page in wiki_pages]

    dictionary = corpora.Dictionary(texts)

    corpus = [dictionary.doc2bow(text) for text in texts]

    bm25_obj = bm25.BM25(corpus)

    dataset = list(jsonlines.open(args.dataset))[1:]    # get rid of the header line

    print("The dataset has {} claims".format(len(dataset)))

    output = jsonlines.open('{}/abstract_retrieval_BM25_top_{}.jsonl'.format(args.output_dir, args.k), 'w')

    page_indices_corpus = []

    for data in tqdm(dataset):

        query = data['claim']

        query_doc = dictionary.doc2bow(query.split())

        scores = np.asarray(bm25_obj.get_scores(query_doc))

        page_indices = scores.argsort()[::-1].tolist()[:args.k]     # get the top k page indices per claim

        # write out the dataset with retrieved pages

        page_id_rank = [wiki_pages[idx]['title'] for idx in page_indices]      # turn the page indices to page titles (real ids)

        data['predicted_pages'] = page_id_rank

        output.write(data)
"""