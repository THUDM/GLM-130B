import sys
sys.path.append("..")

import numpy as np
from argparse import ArgumentParser
import os
import dataloader
from collections import Counter
from collections import defaultdict 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-file", default="../../../data/bias.json", type=str)
    return parser.parse_args()

def main(args):
    filename = args.input_file
    dataset = dataloader.StereoSet(filename, ignore_harm=True)
    
    intrasentence_examples = dataset.get_intrasentence_examples()
    intersentence_examples = dataset.get_intersentence_examples()
    c = defaultdict(lambda: Counter())

    for example in intrasentence_examples:
        c[example.bias_type][example.target] += 1 

    for example in intersentence_examples:
        c[example.bias_type][example.target] += 1 

    for domain, term in c.items():
        print()
        print(domain)
        for k, v in sorted(term.items(), key=lambda x: x[1], reverse=True):
            print(f"{k}: {v}")
    print()
        

if __name__=="__main__":
    args = parse_args()
    main(args)
