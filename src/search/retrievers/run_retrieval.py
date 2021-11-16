import argparse

import ner
import vector_similarity


parser = argparse.ArgumentParser(description='Run retrieval codes.')

parser.add_argument('--ner', action="store_true", help='Use NER matching for section retrieval')
parser.add_argument('--vector', action="store_true", help='Use vector-based similarity matching for section retrieval')
parser.add_argument('--split', action="store_true", help='Retrieve on split claims')

parser.add_argument('--N', type=int, help='Number of sections per claim')
parser.add_argument('--subN', type=int, help='Number of sections per subclaim (use only if running on split claims')


parser.add_argument('--datafile', type=str, help='File where claims:top pages are stored')
parser.add_argument('--indexdir', type=str, help='Directory where inverted index files (if needed) are stored in required format')
parser.add_argument('--writefile', type=str, help='File where claims:retrieved sections is stored as jsonl')
parser.add_argument('--db_path', type=str, help='Path to feverous_wikiv1.db')


args = parser.parse_args()


if __name__ == "__main__":

    # if using NER
    if args.ner:
        print("Function to be reconfigured")
        # ner.run(args.datadir, args.writedir)

    # if using vector-based similarity matching
    elif args.vector and args.split:
        vector_similarity.run_split(args.db_path, args.datafile, args.indexdir, args.writefile, args.N, args.subN)

    # if using vector-based similarity matching
    elif args.vector:
        vector_similarity.run(args.db_path, args.datafile, args.indexdir, args.writefile, args.N)
