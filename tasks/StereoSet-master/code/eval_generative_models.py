import json
import os
from argparse import ArgumentParser
from collections import Counter
from random import shuffle

import numpy as np
import torch
import transformers
from colorama import Back, Fore, Style, init
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import dataloader
from intersentence_loader import IntersentenceDataset
from models import models

init()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained-class", default="gpt2", type=str,
                        help="Choose the pretrained model to load.")
    parser.add_argument("--no-cuda", default=False, action="store_true")
    parser.add_argument("--batch-size", default=10, type=int)
    parser.add_argument("--input-file", default="../data/dev.json",
                        type=str, help="Choose the dataset to evaluate on.")
    parser.add_argument("--output-dir", default="predictions/", type=str,
                        help="Choose the output directory to store predictions in.")
    parser.add_argument("--intrasentence-model",
                        default="GPT2LM", type=str,
                        help="Choose a model architecture for the intrasentence task.")
    parser.add_argument("--intrasentence-load-path", default=None,
                        help="Load a pretrained model for the intrasentence task.")

    parser.add_argument("--intersentence-model",
                        default="ModelNSP", type=str, help="Choose a intersentence model architecture.")
    parser.add_argument("--intersentence-load-path", default=None, 
                        help="Load a pretrained model for the intersentence task.")

    parser.add_argument("--tokenizer", default="GPT2Tokenizer", type=str)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--unconditional_start_token",
                        default="<|endoftext|>", type=str, help="Beginning of sequence token.")
    parser.add_argument("--skip-intersentence",
                        default=False, action="store_true", help="Skip the intersentence task.")
    parser.add_argument("--skip-intrasentence",
                        default=False, action="store_true", help="SKip the intrasentence task.")
    parser.add_argument("--small", default=False, action="store_true")
    return parser.parse_args()


class BiasEvaluator(object):
    def __init__(self, pretrained_class="gpt2", no_cuda=False, batch_size=51, input_file="data/bias.json",
                 intrasentence_model="GPT2LM", intrasentence_load_path=None, intersentence_model="ModelNSP",
                 intersentence_load_path=None, tokenizer="GPT2Tokenizer", unconditional_start_token="<|endoftext|>",
                 skip_intrasentence=False, skip_intersentence=False, max_seq_length=64, small=False,
                 output_dir="predictions/"):
        print(f"Loading {input_file}...")
        self.BATCH_SIZE = batch_size
        filename = os.path.abspath(input_file)
        self.dataloader = dataloader.StereoSet(filename)
        self.cuda = not no_cuda
        self.device = "cuda" if self.cuda else "cpu"
        self.SKIP_INTERSENTENCE = skip_intersentence
        self.SKIP_INTRASENTENCE = skip_intrasentence
        self.UNCONDITIONAL_START_TOKEN = unconditional_start_token

        self.PRETRAINED_CLASS = pretrained_class
        self.TOKENIZER = tokenizer
        self.tokenizer = getattr(transformers, self.TOKENIZER).from_pretrained(
            self.PRETRAINED_CLASS)

        self.INTRASENTENCE_MODEL = intrasentence_model
        self.INTRASENTENCE_LOAD_PATH = intrasentence_load_path
        self.INTERSENTENCE_MODEL = intersentence_model
        self.INTERSENTENCE_LOAD_PATH = intersentence_load_path
        self.max_seq_length = max_seq_length

        print("---------------------------------------------------------------")
        print(
            f"{Fore.LIGHTCYAN_EX}                     ARGUMENTS                 {Style.RESET_ALL}")
        print(
            f"{Fore.LIGHTCYAN_EX}Pretrained class:{Style.RESET_ALL} {pretrained_class}")
        print(f"{Fore.LIGHTCYAN_EX}Unconditional Start Token: {Style.RESET_ALL} {self.UNCONDITIONAL_START_TOKEN}")
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
        print()
        print(
            f"{Fore.LIGHTRED_EX}Evaluating bias on intrasentence tasks...{Style.RESET_ALL}")

        model = getattr(models, self.INTRASENTENCE_MODEL)(
            self.PRETRAINED_CLASS).to(self.device)
        model.eval()

        start_token = torch.tensor(self.tokenizer.encode(
            self.UNCONDITIONAL_START_TOKEN)).to(self.device).unsqueeze(0)
        initial_token_probabilities = model(start_token)
        initial_token_probabilities = torch.softmax(
            initial_token_probabilities[0], dim=-1)

        # ensure that our batch size is 1, and that our initial token isn't split into subwords.
        assert initial_token_probabilities.shape[0] == 1
        assert initial_token_probabilities.shape[1] == 1

        clusters = self.dataloader.get_intrasentence_examples()
        predictions = []
        for cluster in tqdm(clusters):
            for sentence in cluster.sentences:
                probabilities = {}
                tokens = self.tokenizer.encode(sentence.sentence)
                joint_sentence_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()]
                tokens_tensor = torch.tensor(
                    tokens).to(self.device).unsqueeze(0)
                output = torch.softmax(model(tokens_tensor)[0], dim=-1)
                for idx in range(1, len(tokens)):
                    joint_sentence_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                # ensure that we have a probability on every token
                assert len(tokens) == len(joint_sentence_probability)

                score = np.sum([np.log2(i) for i in joint_sentence_probability]) 
                score /= len(joint_sentence_probability)
                score = np.power(2, score)

                probabilities['id'] = sentence.ID
                probabilities['score'] = score

                predictions.append(probabilities)

        return predictions

    def evaluate_intersentence(self):
        model = getattr(models, self.INTERSENTENCE_MODEL)(
            self.PRETRAINED_CLASS).to(self.device)

        if self.PRETRAINED_CLASS == "gpt2-xl":
            model = amp.initialize(model, opt_level="O3")

        start_token = torch.tensor(self.tokenizer.encode(
            self.UNCONDITIONAL_START_TOKEN)).to(self.device).unsqueeze(0)
        initial_token_probabilities = model(start_token)
        initial_token_probabilities = torch.softmax(
            initial_token_probabilities[0], dim=-1)
        assert initial_token_probabilities.shape[0] == 1
        assert initial_token_probabilities.shape[1] == 1

        model.eval()
        clusters = self.dataloader.get_intersentence_examples()[:1000]
        predictions = []

        # iterate over triplets (pro, anti, neg)
        for cluster in tqdm(clusters):
            context = cluster.context

            # iterate over biased sentences
            for sentence in cluster.sentences:
                probabilities = {}
                if context[-1] not in [".", "!", "?"]:
                    context = f"{context}."
                    # context = context[:-1]
                full_sentence = f"{context} {sentence.sentence}"

                probabilities = {}

                tokens = self.tokenizer.encode(full_sentence)
                tokens_tensor = torch.tensor(
                    tokens).to(self.device).unsqueeze(0)

                context_length = len(self.tokenizer.encode(context))

                # gets the probability of the first item in the biased sentence
                sentence_probability = [
                    initial_token_probabilities[0, 0, tokens[context_length]].item()]

                # gets the probability of the first token in the context sentence
                context_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()]

                # sets up the positional tensor
                positions = [
                    0 if idx < context_length else 1 for idx in range(len(tokens))]
                positions_tensor = torch.tensor(
                    positions).unsqueeze(0).to(self.device)

                logits = model(tokens_tensor)

                # we use the 0th item since that corresponds to the prediction scores over vocab tokens
                output = torch.softmax(logits[0], dim=-1)

                # iterate over the context and setup those probabilities.
                for idx in range(1, context_length):
                    # ASSUMPTION: the 0th output corresponds to the probability of the 1st token.
                    context_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                # iterate over the sentence and setup those probabilities.
                for idx in range(1, len(tokens)):
                    # ASSUMPTION: the 0th output corresponds to the probability of the 1st token.
                    sentence_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                full_sentence = f"{sentence.sentence}"
                tokens = self.tokenizer.encode(full_sentence)
                tokens_tensor = torch.tensor(
                    tokens).to(self.device).unsqueeze(0)
                no_context_probability = [
                    initial_token_probabilities[0, 0, tokens[0]].item()]
                logits = model(tokens_tensor)
                output = torch.softmax(logits[0], dim=-1)

                # setup the probability for the sentence if we didn't provide the context
                for idx in range(1, len(tokens)):
                    no_context_probability.append(
                        output[0, idx-1, tokens[idx]].item())

                context_score = np.mean([np.log2(i)
                                         for i in context_probability])

                sentence_score = np.mean([np.log2(i)
                                          for i in sentence_probability])
                no_context_score = np.mean(
                    [np.log2(i) for i in no_context_probability])

                overall_score = no_context_score / context_score
                probabilities['id'] = sentence.ID
                probabilities['score'] = overall_score

                predictions.append(probabilities)
        return predictions

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def evaluate_nsp_intersentence(self):
        print()
        print(
            f"{Fore.LIGHTBLUE_EX}Evaluating bias on intersentence tasks...{Style.RESET_ALL}")
        nsp_dim = 300
        model = getattr(models, self.INTERSENTENCE_MODEL)(
            self.PRETRAINED_CLASS, nsp_dim=nsp_dim).to(self.device)

        if "gpt2" in args.tokenizer.lower():
            print("Adding <PAD> token to tokenizer...")
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            model.core_model.resize_token_embeddings(len(self.tokenizer))

        print(f"Number of parameters: {self.count_parameters(model):,}")
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

        if self.INTERSENTENCE_LOAD_PATH:
            model.load_state_dict(torch.load(self.INTERSENTENCE_LOAD_PATH))

        model.eval()
        dataset = IntersentenceDataset(self.tokenizer, args)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        predictions = []

        for batch_num, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids, token_type_ids, attention_mask, sentence_id = batch
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            outputs = model(input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

            if type(outputs) == tuple:
                outputs = outputs[0]
            outputs = torch.softmax(outputs, dim=1)

            for idx in range(input_ids.shape[0]):
                probabilities = {}
                probabilities['id'] = sentence_id[idx]

                if "bert" in self.PRETRAINED_CLASS:
                    probabilities['score'] = outputs[idx, 0].item()
                else:
                    probabilities['score'] = outputs[idx, 1].item()
                predictions.append(probabilities)

        return predictions

    def evaluate(self):
        bias = {}
        if not self.SKIP_INTRASENTENCE:
            intrasentence_bias = self.evaluate_intrasentence()
            bias['intrasentence'] = intrasentence_bias
        if not self.SKIP_INTERSENTENCE:
            if self.INTERSENTENCE_MODEL == "ModelNSP":
                print("Using NSP evaluation mechanism!")
                intersentence_bias = self.evaluate_nsp_intersentence()
            else:
                intersentence_bias = self.evaluate_intersentence()
            bias['intersentence'] = intersentence_bias
        return bias


if __name__ == "__main__":
    args = parse_args()
    evaluator = BiasEvaluator(**vars(args))
    results = evaluator.evaluate()
    output_file = os.path.join(
        args.output_dir, f"predictions_{args.pretrained_class}_{args.intersentence_model}_{args.intrasentence_model}.json")
    with open(output_file, "w+") as f:
        json.dump(results, f, indent=2)
