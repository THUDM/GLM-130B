import json
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", default="results.json", type=str)
    return parser.parse_args()

def main(args):
    with open(args.results_file, "rb") as f:
        results = json.load(f)
    summaries = {"profession": {"Count": [], "LM Score": [], "SS Score": [], "ICAT Score": []}, "religion": {"Count": [], "LM Score": [], "SS Score": [], "ICAT Score": []}, "race": {"Count": [], "LM Score": [], "SS Score": [], "ICAT Score": []}, "gender": {"Count": [], "LM Score": [], "SS Score": [], "ICAT Score": []}}
    k = results['Ensemble']
    for v in ['intersentence', 'intrasentence']:
        for domain, stats in k[v].items(): 
            if domain=="overall":
                continue
            summaries[domain]['Count'].append(stats['Count'])
            summaries[domain]['LM Score'].append(stats['LM Score'])
            try:
                summaries[domain]['SS Score'].append(stats['SS Score'])
            except KeyError:
                print(stats)
            summaries[domain]['ICAT Score'].append(stats['ICAT Score'])

    print("=========== Computing Per-Domain Statistics ===========") 
    print()
    for k, v in summaries.items():
        print(f"Domain: {k.capitalize()}")
        for stat, score in v.items():
            if stat=="Count":
                print(f"{stat}: {np.mean(score)*2:.1f}")
            else:
                print(f"{stat}: {np.mean(score):.1f}")
        print()

if __name__=="__main__":
    args = parse_args()
    main(args)
