import argparse
import torch
import json
from torch.utils.data import DataLoader
from datasets import concatenate_datasets, Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm
from typing import List


parser = argparse.ArgumentParser()

parser.add_argument('--train', type=str, help="Path to training data file")
parser.add_argument('--save_model', type=str, help="Path to place to save model")
parser.add_argument('--lines', type=int, help="Number of lines to read from training data file")

# args with defaults
parser.add_argument('--lr-base', type=float, default=5e-5)
parser.add_argument('--lr-linear', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch-size-gpu', type=int, default=8, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=128, help='The batch size for each gradient update')

args = parser.parse_args()

ftrain = args.train
fsave = args.save_model
lines = args.lines
lr_base = args.lr_base
lr_linear = args.lr_linear
epochs = args.epochs
batch_size_gpu = args.batch_size_gpu
batch_size_accumulated = args.batch_size_accumulated

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(f'Using device "{device}"')

# ===================== CELL 2 ========================


def evaluate(model, dataset):

    model.eval()

    targets = []

    outputs = []

    with torch.no_grad():

        for batch in DataLoader(dataset, batch_size=batch_size_gpu):

            encoded_dict = encode(batch['claim'], batch['evidence'])
            logits = model(**encoded_dict.to(device)).logits
            targets.extend(batch['label'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())

    return {
        'F1-macro': round(f1_score(targets, outputs, average='macro'), 4)
    }


def mk_datapoints(claim, sections_data):
    """
    Input: claim, {section_id, [[item_id1... item_idN]], label}
    Output: a list of datapoints of the form:
        [{section_id, [item1... itemN], claim, label}]
    """

    datapoints = []

    for point in sections_data:

        section_id = point['section_id']
        # page_title = section_id.split("_")[0]

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


if __name__ == "__main__":

    seed = 123

    dataset, torch_dataset = read_data(ftrain, lines)

    # balance training set across labels
    trainset_y = torch_dataset.filter(lambda example: example['label'] == 1)
    trainset_n = torch_dataset.filter(lambda example: example['label'] == 0)
    print('unbalanced dataset sample counts:', len(trainset_y), len(trainset_n))

    n_y = len(trainset_y)
    trainset_y = Dataset.from_dict(trainset_y.shuffle(seed=seed)[:n_y])
    trainset_n = Dataset.from_dict(trainset_n.shuffle(seed=seed)[:n_y * 4])
    print('balanced dataset sample counts:', len(trainset_y), len(trainset_n))
    trainset_sampled = concatenate_datasets([trainset_y, trainset_n])

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    config = AutoConfig.from_pretrained('roberta-base', num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', config=config).to(device)
    optimizer = torch.optim.Adam([
                    {'params': model.roberta.parameters(), 'lr': lr_base}, 
                    {'params': model.classifier.parameters(), 'lr': lr_linear}])

    for e in range(epochs):

        model.train()

        t = tqdm(DataLoader(trainset_sampled, batch_size=batch_size_gpu, shuffle=True))
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs * batch_size_accumulated)

        for i, batch in enumerate(t):

            encoded_dict = encode(batch['claim'], batch['evidence'])
            outputs = model(**encoded_dict.to(device), labels=batch['label'].long().to(device))

            loss = outputs.loss
            loss.backward()

            if (i + 1) % (batch_size_accumulated // batch_size_gpu) == 0:

                optimizer.step()
                optimizer.zero_grad()

                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')

                scheduler.step()

        # Eval

    #    train_score = evaluate(model, torch_dataset)
    #    print(f'Epoch {e} train score:')
    #    print("Acc:", train_score["Accuracy"], "F1-macro:", train_score["F1-macro"])

    # save model
    model.save_pretrained(fsave)