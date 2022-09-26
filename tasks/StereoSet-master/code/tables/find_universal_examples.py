import sys
sys.path.append("..")

import json
import os
from argparse import ArgumentParser
from collections import Counter
from torch import nn
from collections import defaultdict, Counter

import numpy as np
import torch
from colorama import Back, Fore, Style, init
from tqdm import tqdm

from glob import glob
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import dataloader
from intersentence_loader import IntersentenceDataset
import transformers
from torch.utils.data import DataLoader
from models import models 

init()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", default="dev_predictions/", type=str)
    parser.add_argument("--gold-file", default="final_dataset/dev.json", type=str)
    parser.add_argument("--type", choices=["pro", "anti", "unrelated"], required=True)
    parser.add_argument("--domain-filter", choices=["religion", "profession", "gender", "race"])
    return parser.parse_args()

def main(args):
    MODEL_NAMES = ['bert-large-cased', 'xlnet-large-cased', 'roberta-base', 'gpt2-medium', 'xlnet-base-cased', 'roberta-large', 'gpt2-large', 'bert-base-cased', 'gpt2']
    sentence_ids = [] # a list of tuples of (pro_id, anti_id, unrelated_id)

    gold_file = dataloader.StereoSet(args.gold_file) 
    intrasentence_examples = gold_file.get_intrasentence_examples()
    intersentence_examples = gold_file.get_intersentence_examples()
    examples = intrasentence_examples + intersentence_examples
    target_counts = Counter()

    for example in examples:
        d = {}
        for sentence in example.sentences:
            d[sentence.gold_label] = sentence
        d['type'] = example.bias_type
        d['target'] = example.target
        target_counts[example.target] += 1
        sentence_ids.append(d)

    sent2score = defaultdict(lambda: dict())
    for predictions_file in glob(args.input_dir + "*.json"):
        idx = 2 if "_" in args.input_dir else 1 
        model_name = predictions_file.split("_")[idx]
        with open(predictions_file, "r") as f:
            results = json.load(f) 
        for result in results['intrasentence']:
            sent2score[result['id']][model_name] = result['score'] 
        for result in results['intersentence']:
            sent2score[result['id']][model_name] = result['score'] 

    count = 0.0
    domains = Counter()
    terms_per_domain = defaultdict(lambda: Counter())
    for sentence_pair in sentence_ids: 
        l = []
        for model in MODEL_NAMES:
            # Pro-Stereotype Case
            if args.type=="pro" and ((sent2score[sentence_pair['stereotype'].ID][model] > sent2score[sentence_pair['anti-stereotype'].ID][model]) and (sent2score[sentence_pair['stereotype'].ID][model] > sent2score[sentence_pair['unrelated'].ID][model])):
                l.append(True)
            # anti-stereotype case
            elif args.type=="anti" and ((sent2score[sentence_pair['anti-stereotype'].ID][model] > sent2score[sentence_pair['stereotype'].ID][model]) and (sent2score[sentence_pair['anti-stereotype'].ID][model] > sent2score[sentence_pair['unrelated'].ID][model])):
                l.append(True)
            elif args.type=="unrelated" and ((sent2score[sentence_pair['unrelated'].ID][model] > sent2score[sentence_pair['stereotype'].ID][model]) and (sent2score[sentence_pair['unrelated'].ID][model] > sent2score[sentence_pair['anti-stereotype'].ID][model])):
                l.append(True)
            else:
                l.append(False)
        if all(l):
            for k, v in sentence_pair.items():
                if k in ["type", "target"]:
                    continue
                if args.domain_filter==None or args.domain_filter==sentence_pair['type']:
                    print(f"{k}: {v.sentence}, {v.ID}")
            print()
            count += 1.0
            domains[sentence_pair['type']] += 1
            terms_per_domain[sentence_pair['type']][sentence_pair['target']] += 1

    print(f"Number of clusters that models agree on: {count}")
    print("Breakdown by Domain:", domains)
    for domain in domains.keys():
        print(f"Domain: {domain}")
        terms = terms_per_domain[domain]
        normalized_terms = {}
        for k, v in terms.items():
            normalized_terms[k] = v / target_counts[k]
        normalized_terms = {k: v for k, v in sorted(normalized_terms.items(), key=lambda item: item[1], reverse=True)}
        print(normalized_terms)
        print()

if __name__ == "__main__":
    args = parse_args()
    main(args) 
