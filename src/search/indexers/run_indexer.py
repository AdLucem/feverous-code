import argparse

from vector_indexer import run_vector_indexer


parser = argparse.ArgumentParser(description='Run indexer codes.')
parser.add_argument('--tfidf', action="store_true", help='Run TF-IDF indexing')
parser.add_argument('--vector', action="store_true", help='Run vector-based indexing')

parser.add_argument('--datafile', type=str, help='File where data to be read is stored in required format')
parser.add_argument('--index_dir', type=str, help='Dir where index is stored')
parser.add_argument('--db_path', type=str, help="Path to feverous_wikiv1.db")


args = parser.parse_args()


if __name__ == "__main__":

    # if using NER
    if args.tfidf:
        print("This indexer is yet to be attached to the main file")
    elif args.vector:
        run_vector_indexer(args.datafile, args.index_dir, args.db_path)
