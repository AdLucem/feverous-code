import os
from scipy.stats import pearsonr


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


def get_table_cells(annotation):
    """Get table cell evidence for annotation (get first evidence
    set that is table-only)"""

    evidence = annotation.get_evidence()
    evtype = annotation.get_evidence_type()

    if len(evtype) > 1:
        idx = evtype.index(EvidenceType['TABLE'])
        return evidence[idx]
    else:
        return evidence[0]


def main():
    input_path = "/home/kw/Data/FEVEROUS"
    annotations = list(AnnotationProcessor(
                       os.path.join(input_path, 'train.jsonl')))
    db = FeverousDB("/home/kw/Data/FEVEROUS/feverous_wikiv1.db")

    table_annotations = filter_E_cells(annotations)

    sum_pno = n = 0
    for annotation in table_annotations:
        # print(annotation.get_claim())
        # verdict = annotation.get_verdict()
        # print(verdict)

        cells = get_table_cells(annotation)

        cell_tables = [cell.split("_")[0] for cell in cells]
        pages = list(set(cell_tables))

        n += 1
        sum_pno += len(pages)

    print(sum_pno, n, sum_pno/n)

"""
        # split into {page: [(table_id, cell_i, cell_j)]}
        d = {}
        for page in pages:
            d[page] = []
        for cell in cells:
            if 'caption' in cell:
                pass
            elif 'header' in cell:
                s = cell.split("_")
                d[s[0]].append((cell, s[3], s[4], s[5]))
            else:
                s = cell.split("_")
                d[s[0]].append((cell, s[2], s[3], s[4]))

        for page in d:

            page_json = db.get_doc_json(page)
            wiki_page = WikiPage(page, page_json)
            wiki_tables = wiki_page.get_tables()

            cell, table_id, row, col = d[page][0]
            table = wiki_tables[int(table_id)]
            rows = table.get_rows()

            for i in range(len(rows)):
                row = rows[i].get_row_cells()
                repr = '|'.join([str(cell) for cell in row])
                print(repr)
            print("---------------------------------")

        print("=============================================")
"""

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