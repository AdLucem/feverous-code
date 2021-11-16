import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import time  

n = 1000

emma = nltk.corpus.gutenberg.sents('austen-emma.txt')

texts = []
for sent in emma[:n]:
    texts.append(" ".join(sent))

t0 = time.time()
for text in texts:
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform([text])
t1 = time.time()

print(t1 - t0)
