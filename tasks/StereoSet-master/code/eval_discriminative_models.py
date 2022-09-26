import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import cpu_count

import numpy as np
import torch
import transformers
from colorama import Fore, Style, init
from joblib import Parallel, delayed
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader
from intersentence_loader import IntersentenceDataset
from models import models

init()


def parse_args():
    """ Parses the command line arguments. """
    pretrained_model_choices = ['bert-base-uncased', 'bert-base-cased', "bert-large-uncased-whole-word-masking",
                                'bert-large-uncased', 'bert-large-cased', 'gpt2', 'gpt2-medium', 'gpt2-large', 'roberta-base',
                                'roberta-large', 'xlnet-base-cased', 'xlnet-large-cased']
    tokenizer_choices = ["RobertaTokenizer", "BertTokenizer", "XLNetTokenizer"]
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained-class", default="bert-base-cased", choices=pretrained_model_choices,
        help="Choose the pretrained model to load from.")
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
    parser.add_argument("--intrasentence-model", type=str, default='BertLM', choices=[
                        'BertLM', 'BertNextSentence', 'RoBERTaLM', 'XLNetLM', 'XLMLM', 'GPT2LM', 'ModelNSP'],
                        help="Choose a model architecture for the intrasentence task.")
    parser.add_argument("--intrasentence-load-path", default=None,
                        help="Load a pretrained model for the intrasentence task.")
    parser.add_argument("--skip-intersentence",
                        default=False, action="store_true", help="Skip intersentence evaluation.")
    parser.add_argument("--intersentence-model", type=str, default='BertNextSentence', choices=[
                        'BertLM', 'BertNextSentence', 'RoBERTaLM', 'XLNetLM', 'XLMLM', 'GPT2LM', 'ModelNSP'],
                        help="Choose the model for the intersentence task.")
    parser.add_argument("--intersentence-load-path", default=None,
                        help="Path to the pretrained model for the intersentence task.")
    parser.add_argument("--tokenizer", type=str,
                        default='BertTokenizer', choices=tokenizer_choices,
                        help="Choose a string tokenizer.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=128)
    return parser.parse_args()


class BiasEvaluator():
    def __init__(self, pretrained_class="bert-large-uncased-whole-word-masking", no_cuda=False,
                 input_file="data/bias.json", intrasentence_model="BertLM",
                 intersentence_model="BertNextSentence", tokenizer="BertTokenizer",
                 intersentence_load_path=None, intrasentence_load_path=None, skip_intrasentence=False,
                 skip_intersentence=False, batch_size=1, max_seq_length=128, 
                 output_dir="predictions/", output_file="predictions.json"):
        print(f"Loading {input_file}...")
        filename = os.path.abspath(input_file)
        self.dataloader = dataloader.StereoSet(filename)
        self.cuda = not no_cuda
        self.device = "cuda" if self.cuda else "cpu"

        self.INTRASENTENCE_LOAD_PATH = intrasentence_load_path
        self.INTERSENTENCE_LOAD_PATH = intersentence_load_path
        self.SKIP_INTERSENTENCE = skip_intersentence
        self.SKIP_INTRASENTENCE = skip_intrasentence
        self.INTRASENTENCE_LOAD_PATH = intrasentence_load_path
        self.INTERSENTENCE_LOAD_PATH = intersentence_load_path

        self.PRETRAINED_CLASS = pretrained_class
        self.TOKENIZER = tokenizer
        self.tokenizer = getattr(transformers, self.TOKENIZER).from_pretrained(
            self.PRETRAINED_CLASS, padding_side="right")

        # to keep padding consistent with the other models -> improves LM score.
        if self.tokenizer.__class__.__name__ == "XLNetTokenizer":
            self.tokenizer.padding_side = "right"
        self.MASK_TOKEN = self.tokenizer.mask_token

        # Set this to be none if you don't want to batch items together!
        self.batch_size = batch_size
        self.max_seq_length = None if self.batch_size == 1 else max_seq_length

        self.MASK_TOKEN_IDX = self.tokenizer.encode(
            self.MASK_TOKEN, add_special_tokens=False)
        assert len(self.MASK_TOKEN_IDX) == 1
        self.MASK_TOKEN_IDX = self.MASK_TOKEN_IDX[0]

        self.INTRASENTENCE_MODEL = intrasentence_model
        self.INTERSENTENCE_MODEL = intersentence_model

        print("---------------------------------------------------------------")
        print(
            f"{Fore.LIGHTCYAN_EX}                     ARGUMENTS                 {Style.RESET_ALL}")
        print(
            f"{Fore.LIGHTCYAN_EX}Pretrained class:{Style.RESET_ALL} {pretrained_class}")
        print(f"{Fore.LIGHTCYAN_EX}Mask Token:{Style.RESET_ALL} {self.MASK_TOKEN}")
        print(f"{Fore.LIGHTCYAN_EX}Tokenizer:{Style.RESET_ALL} {tokenizer}")
        print(
            f"{Fore.LIGHTCYAN_EX}Skip Intrasentence:{Style.RESET_ALL} {self.SKIP_INTRASENTENCE}")
        print(
            f"{Fore.LIGHTCYAN_EX}Intrasentence Model:{Style.RESET_ALL} {self.INTRASENTENCE_MODEL}")
        print(
            f"{Fore.LIGHTCYAN_EX}Skip Intersentence:{Style.RESET_ALL} {self.SKIP_INTERSENTENCE}")
        print(
            f"{Fore.LIGHTCYAN_EX}Intersentence Model:{Style.RESET_ALL} {self.INTERSENTENCE_MODEL}")
        print(f"{Fore.LIGHTCYAN_EX}CUDA:{Style.RESET_ALL} {self.cuda}")
        print("---------------------------------------------------------------")

    def evaluate_intrasentence(self):
        model = getattr(models, self.INTRASENTENCE_MODEL)(
            self.PRETRAINED_CLASS).to(self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.eval()

        print()
        print(
            f"{Fore.LIGHTRED_EX}Evaluating bias on intrasentence tasks...{Style.RESET_ALL}")

        if self.INTRASENTENCE_LOAD_PATH:
            state_dict = torch.load(self.INTRASENTENCE_LOAD_PATH)
            model.load_state_dict(state_dict)

        pad_to_max_length = True if self.batch_size > 1 else False
        dataset = dataloader.IntrasentenceLoader(self.tokenizer, max_seq_length=self.max_seq_length,
                                                 pad_to_max_length=pad_to_max_length, 
                                                 input_file=args.input_file)

        loader = DataLoader(dataset, batch_size=self.batch_size)
        word_probabilities = defaultdict(list)

        # calculate the logits for each prediction
        for sentence_id, next_token, input_ids, attention_mask, token_type_ids in tqdm(loader, total=len(loader)):
            # start by converting everything to a tensor
            input_ids = torch.stack(input_ids).to(self.device).transpose(0, 1)
            attention_mask = torch.stack(attention_mask).to(
                self.device).transpose(0, 1)
            next_token = next_token.to(self.device)
            token_type_ids = torch.stack(token_type_ids).to(
                self.device).transpose(0, 1)

            mask_idxs = (input_ids == self.MASK_TOKEN_IDX)

            # get the probabilities
            output = model(input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)[0].softmax(dim=-1)

            output = output[mask_idxs]
            output = output.index_select(1, next_token).diag()
            for idx, item in enumerate(output):
                word_probabilities[sentence_id[idx]].append(item.item())

        # now reconcile the probabilities into sentences
        sentence_probabilties = []
        for k, v in word_probabilities.items():
            pred = {}
            pred['id'] = k
            # score = np.sum([np.log2(i) for i in v]) + np.log2(len(v))
            score = np.mean(v)
            pred['score'] = score
            sentence_probabilties.append(pred)

        return sentence_probabilties

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_intersentence(self):
        print()
        print(
            f"{Fore.LIGHTBLUE_EX}Evaluating bias on intersentence tasks...{Style.RESET_ALL}")
        model = getattr(models, self.INTERSENTENCE_MODEL)(
            self.PRETRAINED_CLASS).to(self.device)

        print(f"Number of parameters: {self.count_parameters(model):,}")
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

        if self.INTERSENTENCE_LOAD_PATH:
            model.load_state_dict(torch.load(self.INTERSENTENCE_LOAD_PATH))

        model.eval()
        dataset = IntersentenceDataset(self.tokenizer, args)
        # TODO: test this on larger batch sizes.
        assert args.batch_size == 1
        dataloader = DataLoader(dataset, shuffle=True, num_workers=0)

        if args.no_cuda:
            n_cpus = cpu_count()
            print(f"Using {n_cpus} cpus!")
            predictions = Parallel(n_jobs=n_cpus, backend="multiprocessing")(delayed(process_job)(
                batch, model, self.PRETRAINED_CLASS) for batch in tqdm(dataloader, total=len(dataloader)))
        else:
            predictions = []

            for batch_num, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                input_ids, token_type_ids, attention_mask, sentence_id = batch
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = model(input_ids, token_type_ids=token_type_ids)
                if type(outputs) == tuple:
                    outputs = outputs[0]
                outputs = torch.softmax(outputs, dim=1)

                for idx in range(input_ids.shape[0]):
                    probabilities = {}
                    probabilities['id'] = sentence_id[idx]
                    if "bert" == self.PRETRAINED_CLASS[:4] or "roberta-base" == self.PRETRAINED_CLASS:
                        probabilities['score'] = outputs[idx, 0].item()
                    else:
                        probabilities['score'] = outputs[idx, 1].item()
                    predictions.append(probabilities)

        return predictions

    def evaluate(self):
        bias = {}
        if not self.SKIP_INTERSENTENCE:
            intersentence_bias = self.evaluate_intersentence()
            bias['intersentence'] = intersentence_bias

        if not self.SKIP_INTRASENTENCE:
            intrasentence_bias = self.evaluate_intrasentence()
            bias['intrasentence'] = intrasentence_bias
        return bias


def process_job(batch, model, pretrained_class):
    input_ids, token_type_ids, sentence_id = batch
    outputs = model(input_ids, token_type_ids=token_type_ids)
    if type(outputs) == tuple:
        outputs = outputs[0]
    outputs = torch.softmax(outputs, dim=1)

    pid = sentence_id[0]
    # if "bert"==self.PRETRAINED_CLASS[:4]:
    if "bert" in pretrained_class:
        pscore = outputs[0, 0].item()
    else:
        pscore = outputs[0, 1].item()
    return (pid, pscore)


if __name__ == "__main__":
    args = parse_args()
    evaluator = BiasEvaluator(**vars(args))
    results = evaluator.evaluate()
    if args.output_file is not None:
        output_file = args.output_file
    else:
        output_file = f"predictions_{args.pretrained_class}_{args.intersentence_model}_{args.intrasentence_model}.json"

    output_file = os.path.join(args.output_dir, output_file)
    with open(output_file, "w+") as f:
        json.dump(results, f, indent=2)
