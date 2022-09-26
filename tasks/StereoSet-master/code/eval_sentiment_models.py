import sys
import json
import os
from argparse import ArgumentParser

import numpy as np
import spacy
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertForMaskedLM, BertModel, BertTokenizer, BertForSequenceClassification 
from colorama import Fore, Style, init

from intersentence_loader import SentimentIntersentenceDataset  
from dataloader import SentimentIntrasentenceLoader, StereoSet
import utils

nlp = spacy.load('en')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--no-cuda", default=False, action="store_true")
    parser.add_argument(
        "--input-file", default="../data/dev.json", type=str,
        help="Choose the dataset to evaluate on.")
    parser.add_argument("--output-dir", default="predictions/", type=str,
                        help="Choose the output directory for predictions.")
    parser.add_argument("--output-file", default=None, type=str,
                        help="Choose the name of the predictions file")

    parser.add_argument("--skip-intrasentence", help="Skip intrasentence evaluation.",
                        default=False, action="store_true")
    parser.add_argument("--load-path", default="best_models/SentimentBert.pth", type=str,
                        help="Load a pretrained sentiment model.")
    parser.add_argument("--skip-intersentence",
                        default=False, action="store_true", help="Skip intersentence evaluation.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=128)
    return parser.parse_args()


class BiasEvaluator():
    def __init__(self, no_cuda=False, input_file="data/bias.json", skip_intrasentence=False, 
                 skip_intersentence=False, batch_size=1, max_seq_length=128, output_dir="predictions/", 
                 output_file="predictions.json", load_path="best_models/SentimentBert.pth"):
        print(f"Loading {input_file}...")
        filename = os.path.abspath(input_file)
        self.dataloader = StereoSet(filename)
        self.cuda = not no_cuda
        self.device = "cuda" if self.cuda else "cpu"

        self.LOAD_PATH = load_path
        self.SKIP_INTERSENTENCE = skip_intersentence
        self.SKIP_INTRASENTENCE = skip_intrasentence
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # to keep padding consistent with the other models -> improves LM score.
        if self.tokenizer.__class__.__name__ == "XLNetTokenizer":
            self.tokenizer.padding_side = "right"

        # Set this to be none if you don't want to batch items together!
        self.batch_size = batch_size
        self.max_seq_length = None if self.batch_size == 1 else max_seq_length

        print("---------------------------------------------------------------")
        print(
            f"{Fore.LIGHTCYAN_EX}                     ARGUMENTS                 {Style.RESET_ALL}")
        print(
            f"{Fore.LIGHTCYAN_EX}Pretrained Model:{Style.RESET_ALL} {self.LOAD_PATH}")
        print(
            f"{Fore.LIGHTCYAN_EX}Skip Intrasentence:{Style.RESET_ALL} {self.SKIP_INTRASENTENCE}")
        print(
            f"{Fore.LIGHTCYAN_EX}Skip Intersentence:{Style.RESET_ALL} {self.SKIP_INTERSENTENCE}")
        print(
            f"{Fore.LIGHTCYAN_EX}Batch Size:{Style.RESET_ALL} {self.batch_size}")
        print(
            f"{Fore.LIGHTCYAN_EX}Max Seq Length:{Style.RESET_ALL} {self.max_seq_length}")
        print(f"{Fore.LIGHTCYAN_EX}CUDA:{Style.RESET_ALL} {self.cuda}")
        print("---------------------------------------------------------------")

    def evaluate_intrasentence(self):
        print()
        print(
            f"{Fore.LIGHTRED_EX}Evaluating bias on intrasentence tasks...{Style.RESET_ALL}")
        dataset = SentimentIntrasentenceLoader(self.tokenizer, max_seq_length=args.max_seq_length, pad_to_max_length=True, input_file=args.input_file)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)
        num_labels = 2

        model = utils.BertForSequenceClassification(num_labels)
        device = torch.device("cuda" if not args.no_cuda else "cpu")
        print(f"Number of parameters: {self.count_parameters(model):,}")

        model.to(device).eval()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(self.LOAD_PATH))
        self.model = model

        
        bias_predictions = [] 
        for batch_num, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            sentence_id, input_ids, attention_mask, token_type_ids = batch 
            input_ids = input_ids.to(self.device).squeeze(dim=1) 
            attention_mask = attention_mask.to(self.device).squeeze(dim=1) 
            token_type_ids = token_type_ids.to(self.device).squeeze(dim=1) 

            predictions = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            predictions = predictions.softmax(dim=1)
            for idx, prediction in enumerate(predictions[:, 0]):
                score = {"id": sentence_id[idx], "score": prediction.item()}
                bias_predictions.append(score)

        return bias_predictions


    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_intersentence(self):
        print()
        print(
            f"{Fore.LIGHTBLUE_EX}Evaluating bias on intersentence tasks...{Style.RESET_ALL}")
        dataset = SentimentIntersentenceDataset(self.tokenizer, args)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=5)
        num_labels = 2

        model = utils.BertForSequenceClassification(num_labels)
        device = torch.device("cuda" if not args.no_cuda else "cpu")
        print(f"Number of parameters: {self.count_parameters(model):,}")

        model.to(device).eval()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(self.LOAD_PATH))
        self.model = model

        bias_predictions = [] 
        for batch_num, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            sentence_id, input_ids, attention_mask, token_type_ids = batch 
            input_ids = input_ids.to(self.device).squeeze(dim=1) 
            attention_mask = attention_mask.to(self.device).squeeze(dim=1) 
            token_type_ids = token_type_ids.to(self.device).squeeze(dim=1) 

            predictions = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            # print(predictions)
            predictions = predictions.softmax(dim=1)
            for idx, prediction in enumerate(predictions[:, 0]):
                score = {"id": sentence_id[idx], "score": prediction.item()}
                bias_predictions.append(score)

        return bias_predictions

    def evaluate(self):
        bias = {}
        if not self.SKIP_INTERSENTENCE:
            intersentence_bias = self.evaluate_intersentence()
            bias['intersentence'] = intersentence_bias

        if not self.SKIP_INTRASENTENCE:
            intrasentence_bias = self.evaluate_intrasentence()
            bias['intrasentence'] = intrasentence_bias
        return bias


if __name__ == "__main__":
    args = parse_args()
    evaluator = BiasEvaluator(**vars(args))
    results = evaluator.evaluate()

    if args.output_file is not None:
        output_file = args.output_file
    else:
        output_file = f"predictions_SentimentModel.json"

    output_file = os.path.join(args.output_dir, output_file)
    with open(output_file, "w+") as f:
        json.dump(results, f, indent=2)
