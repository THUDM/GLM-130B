import sys
sys.path.append("..")
import numpy as np
from argparse import ArgumentParser
import os
import dataloader
from collections import Counter, defaultdict

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--no-cuda", default=False, action="store_true")
    parser.add_argument("--input-file", default="../../data/bias.json", type=str)
    return parser.parse_args()

def main(args):
    filename = args.input_file
    dataset = dataloader.StereoSet(filename)

    intrasentence_examples = dataset.get_intrasentence_examples()
    intersentence_examples = dataset.get_intersentence_examples()
    c = Counter()

    intrasentence = defaultdict(lambda: [])
    intrasentence_harm = {"neutral": 0, "stereotype": 0, "anti-stereotype": 0, "undecided": 0}

    terms = {"intersentence": defaultdict(lambda: set()), "intrasentence": defaultdict(lambda: set()), "overall": set()}
    cats = {"intersentence": defaultdict(lambda: 0), "intrasentence": defaultdict(lambda: 0), "overall": 0}
    domains_counter = Counter()

    for example in intrasentence_examples:
        terms['intrasentence'][example.bias_type].add(example.target)
        terms['overall'].add(example.target)
        c[example.bias_type] += 1
        cats['overall'] += 1
        cats['intrasentence'][example.bias_type] += 1
        for sentence in example.sentences:
            # intrasentence[sentence.gold_label].append(sentence.sentence)
            intrasentence[example.bias_type].append(sentence.sentence)
        intrasentence_harm[example.harm['gold_label']] += 1

    intersentence = defaultdict(lambda: [])
    intersentence_harm = {"neutral": 0, "stereotype": 0, "anti-stereotype": 0, "undecided": 0}

    for example in intersentence_examples:
        context = example.context
        terms['intersentence'][example.bias_type].add(example.target)
        terms['overall'].add(example.target)
        cats['intersentence'][example.bias_type] += 1
        cats['overall'] += 1
        c[example.bias_type] += 1
        for sentence in example.sentences:
           # intersentence[sentence.gold_label].append((context, sentence.sentence))
           intersentence[example.bias_type].append((context, sentence.sentence))
        intersentence_harm[example.harm['gold_label']] += 1

    print("Intrasentence!")
    lengths = {"intersentence": [], "intrasentence": []} 
    for k, v in intrasentence.items():
        avg_len = np.mean([len(i.split(" ")) for i in v])
        print(f"Average length of {k}: ", avg_len, "words")
        lengths['intrasentence'].append(avg_len)

        # with open(f"corpus/intrasentence_{k}.txt", "w+") as f:
            # f.write("\n".join(v))
    # print(intrasentence_harm)
    print(np.mean(lengths['intrasentence']))
    print()
    print("Intersentence!")
    for k, v in intersentence.items():
        avg_len = np.mean([len(" ".join(i).split(" ")) for i in v])
        print(f"Average length of {k}: ", avg_len, "words")
        lengths['intersentence'].append(avg_len)
        # with open(f"corpus/intersentence_{k}.txt", "w+") as f:
            # f.write("\n".join([f"{i[0]} {i[1]}" for i in v]))
    # print(intersentence_harm)
    print(np.mean(lengths['intersentence']))
    print("Overall Avg Length:", np.mean(lengths['intersentence'] + lengths['intrasentence']))
    print()

    total = sum(c.values())
    print(f"Total Examples: {total}")
    print(f"Number of total terms: {len(terms)}")
    for k, v in sorted(c.items(), key=lambda x: x[0]):
        print(f"{k}: {v}, {v / total}")
    print()

    print("------- TERMS ANALYSIS -------")
    for cat in ['intersentence', 'intrasentence']:
        total = 0
        for domain, s in terms[cat].items():
            print(f"{domain}: {len(s)}")
            total += len(s)
        print(f"{cat.capitalize()}: {total}")
        print()
    print("Overall total:", len(terms['overall']))
    print()

    print("------- TRIPLETS ANALYSIS -------")
    for cat in ['intersentence', 'intrasentence']:
        total = 0
        for domain, s in cats[cat].items():
            print(f"{domain}: {s}")
            total += s
        print(f"{cat.capitalize()}: {total}")
        print()
    print("Overall total:", cats['overall'])
    print()


if __name__=="__main__":
    args = parse_args()
    main(args)
