import argparse
from collections import defaultdict, Counter
from glob import glob
import os
import dataloader
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()

def main(args):
    model_predictions = defaultdict(lambda: {})
    predictions_dir = args.predictions_dir 
    if args.predictions_dir[-1]!="/":
        predictions_dir = args.predictions_dir + "/"

    for model_file in glob(predictions_dir + "*.json"): 
        print()
        print(f"Ingesting {model_file}...")
        with open(model_file, "r+") as f:
            model_preds = json.load(f)

        id2score = {} 
        for p in model_preds['intersentence'] + model_preds['intrasentence']:
            id2score[p['id']] = p['score'] 

        intersentence_ids = set()
        for p in model_preds['intersentence']:
            intersentence_ids.add(p['id'])

        pretrained_class = os.path.basename(model_file).split("_")[1]
        model_predictions[pretrained_class] = id2score

    predictions = Counter()
    stereoset = dataloader.StereoSet(args.gold_file)
    examples = stereoset.get_intrasentence_examples() + stereoset.get_intersentence_examples()
    unrelateds = set()

    BERT_INTERSENTENCE_WEIGHT = 35.0
    GPT_INTERSENTENCE_WEIGHT = 15.0
    BERT_INTRASENTENCE_WEIGHT = 1.0 
    GPT_INTRASENTENCE_WEIGHT = 10000.0 


    for example in examples:
        assert len(example.sentences)==3
        for (pair_a, pair_b) in [(0,1), (1,2), (2,0)]:
            for k in ['gpt2-large', "bert-large-cased", "gpt2-medium"]: 
                v = model_predictions[k]
                id_a = example.sentences[pair_a].ID
                id_b = example.sentences[pair_b].ID

                for pair_x, id_x in [(pair_a, id_a), (pair_b, id_b)]:
                    if example.sentences[pair_x].gold_label=="unrelated":
                        unrelateds.add(id_x)

                prediction_a = v[id_a]
                prediction_b = v[id_b]

                if id_a not in predictions:
                    predictions[id_a] = 0
                if id_b not in predictions:
                    predictions[id_b] = 0

                if id_a in intersentence_ids:
                    if prediction_a==prediction_b:
                        pass
                    elif prediction_a > prediction_b: 
                        if 'gpt2' in k: 
                            predictions[id_a] += GPT_INTERSENTENCE_WEIGHT  * (prediction_a) 
                        else:
                            predictions[id_a] += BERT_INTERSENTENCE_WEIGHT * (prediction_a) 
                    else:
                        if 'gpt2' in k: 
                            predictions[id_b] += GPT_INTERSENTENCE_WEIGHT * (prediction_b)
                        else:
                            predictions[id_b] += BERT_INTERSENTENCE_WEIGHT * (prediction_b)
                else:
                    if prediction_a==prediction_b:
                        pass
                    elif prediction_a > prediction_b: 
                        if 'gpt2' in k: 
                            predictions[id_a] += GPT_INTRASENTENCE_WEIGHT * (prediction_a) 
                        else:
                            predictions[id_a] += BERT_INTRASENTENCE_WEIGHT * (prediction_a) 
                    else:
                        if 'gpt2' in k: 
                            predictions[id_b] += GPT_INTRASENTENCE_WEIGHT * (prediction_b)
                        else:
                            predictions[id_b] += BERT_INTRASENTENCE_WEIGHT * (prediction_b)


    final_predictions = {"intersentence": [], "intrasentence": []}
    for k, v in predictions.items():
        d = {}
        d['id'] = k
        d['score'] = v
        if d['id'] in intersentence_ids:
            final_predictions['intersentence'].append(d)
        else:
            final_predictions['intrasentence'].append(d)

    print("Dumping results to", args.output_file)
    with open(args.output_file, "w+") as f:
        json.dump(final_predictions, f, indent=2)

if __name__=="__main__":
    args = parse_args()
    main(args)
