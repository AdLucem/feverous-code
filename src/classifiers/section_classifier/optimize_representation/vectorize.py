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
