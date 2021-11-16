import argparse
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
from typing import List


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help="Path to trained model")
parser.add_argument('--lines', type=int, help="Number of lines to read from training data file")
parser.add_argument('--dev', type=str, help="Path to dev sections representation")
parser.add_argument("--save", type=str, help="dir to save everything to")

args = parser.parse_args()

fdev = args.dev
lines = args.lines
model_dir = args.model
batch_size_gpu = 8
batch_size_accumulated = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'Using device "{device}"')


# ======================= FUNCTIONS ======================


def evaluate(model, dataset, dataset_json):

    model.eval()

    targets = []

    outputs = []

    with torch.no_grad():

        for batch in tqdm(DataLoader(dataset, batch_size=batch_size_gpu)):

            encoded_dict = encode(batch['claim'], batch['evidence'])
            logits = model(**encoded_dict.to(device)).logits
            targets.extend(batch['label'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())

    with open(args.save + 'dev_data_run.json', "w+") as f:
        json.dump(dataset_json, f)

    with open(args.save + "outputs.json", "w+") as f:
        json.dump(outputs, f)

    scores = {
        'Precision': round(precision_score(targets, outputs), 4),
        'Recall': round(recall_score(targets, outputs), 4)
    }

    with open(args.save + "scores.json", "w+") as f:
        json.dump(scores, f)

    return scores


def mk_datapoints(claim, sections_data):
    """
    Input: claim, {section_id, [[item_id1... item_idN]], label}
    Output: a list of datapoints of the form:
        [{section_id, [item1... itemN], claim, label}]
    """

    datapoints = []

    for point in sections_data:

        section_id = point['section_id']

        label = point["label"]
        sequences = point["sequences"]

        for sequence in sequences:

            datapoints.append({"section_id": section_id,
                               "claim": claim,
                               "sequence": sequence,
                               "label": label})

    return datapoints
        

def read_data(filename, lines):

    data = []
    with open(filename) as f:

        maybe_line = f.readline()

        for i in range(lines):
            line = json.loads(maybe_line)
            
            claim = line["claim"]
            sections_data = line["sections_data"]
            
            data += mk_datapoints(claim, sections_data)

    # make data readable by `datasets`
    claims = [point['claim'] for point in data]
    sequence = ["|".join(point['sequence']) for point in data]
    labels = [point['label'] for point in data]

    data_ = {'claim': claims,
             'evidence': sequence,
             'label': labels}
    dataset = Dataset.from_dict({'claim': claims,
                                 'evidence': sequence,
                                 'label': labels})
    return data_, dataset


def encode(claim: List[str], rationale: List[str]):

    encodings = tokenizer(claim, rationale, padding=True, truncation=True, max_length=512, return_tensors="pt")

    return encodings


# ====================== MAIN ========================


with torch.no_grad():

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    config = AutoConfig.from_pretrained(model_dir, num_labels=2)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config).to(device)

    dev, torch_dev = read_data(fdev, lines)

    train_score = evaluate(model, torch_dev, dev)
    print('Test score:')
    print("Precision:", train_score["Precision"], "Recall:", train_score["Recall"])
