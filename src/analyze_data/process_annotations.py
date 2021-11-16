import os

from utils.annotation_processor import AnnotationProcessor, EvidenceType
from database.feverous_db import FeverousDB
from utils.wiki_page import WikiPage


def filter_E_cells(annotations):
    table_annotations = []

    for ann in annotations:
        evtype = ann.get_evidence_type()

        if EvidenceType['TABLE'] in evtype:
            table_annotations.append(ann)

    return table_annotations


def get_context_cells(annotation):
    """Get contexts for each table cell in each evidence set
    in a particular annotation"""

    contexts = annotation.get_context()

    for evidence_set in contexts:
        print(evidence_set)
        print("--------------------------------------")
    print("============================================")


def main():
    input_path = "/home/kw/Data/FEVEROUS"
    annotations = list(AnnotationProcessor(
                       os.path.join(input_path, 'train.jsonl')))
    # db = FeverousDB("/home/kw/Data/FEVEROUS/feverous_wikiv1.db")

    table_annotations = filter_E_cells(annotations)

    for annotation in table_annotations[:10]:
        contexts = annotation.get_context()
        for context in contexts:
            for key in context:
                print(annotation.get_context_content())


if __name__ == "__main__":
    main()
"""
page_json = db.get_doc_json("Anarchism")
wiki_page = WikiPage("Anarchism", page_json)
wiki_tables = wiki_page.get_tables()
# return list of all Wiki Tables

wiki_table_0 = wiki_tables[0]
wiki_table_0_rows = wiki_table_0.get_rows()
# return list of WikiRows
wiki_table_0_header_rows = wiki_table_0.get_header_rows()
# return list of WikiRows that are headers
is_header_row = wiki_table_0_rows[0].is_header_row()
# or check the row directly whether it is a header


for i in range(len(wiki_table_0_rows)):
    cells_row_0 = wiki_table_0_rows[i].get_row_cells()
    print(len(cells_row_0))
    # return list with WikiCells for row 0
    row_representation = '|'.join([str(cell) for cell in cells_row_0])
    # get cell content seperated by vertical line
    print(row_representation)
row_representation_same = str(cells_row_0)
# or just stringfy the row directly.

#returns WikiTable from Cell_id. Useful for retrieving associated Tables for cell annotations.
table_0_cell_dict = wiki_page.get_table_from_cell_id(cells_row_0[0].get_id())
"""