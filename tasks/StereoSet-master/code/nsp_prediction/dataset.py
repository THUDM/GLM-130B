import glob
import json
import nltk
import numpy as np
import re
from pprint import pprint
from scipy import stats
from tqdm import tqdm
from joblib import Parallel, delayed, dump
from multiprocessing import cpu_count
from torch.utils.data import Dataset
import random
from math import ceil 

class NextSentenceDataset(Dataset):
    def __init__(self, directory, tokenizer, data_frac=1.0, max_seq_length=512, test=False, skip_frac=0.003):
        """
        Args:
            - Directory: directory where the Wikipedia extract is located.
            - Tokenizer: A Huggingface tokenizer to preprocess the text with.
            - Dara Frac: which portion of Wikipedia's data dump to use. We used 10%.
            - Max Sequence Length: maximum sequence length for tokenization and padding.
            - Test: Sample from the end of the file list, if you didn't train on the entire file list.
            - Skip Frac: start from an offset of the data if fine-tuning a pretrained NSP head.
        """
        self.tokenizer = tokenizer
        file_list = glob.glob(f"{directory}/*/wiki_**", recursive=True)
        offset = ceil(len(file_list) * skip_frac)
        if test:
            file_list = file_list[-ceil(len(file_list) * data_frac):] 
        else:
            file_list = file_list[offset:offset+ceil(len(file_list) * data_frac)] 
        lines = []
        self.sentences = Parallel(n_jobs=30, backend="multiprocessing", verbose=1)(delayed(self._process_file)(i) for i in file_list) 
        self.max_seq_length = max_seq_length
        random.seed(9)
        self.memo = []
        self.examples = []
        self.lengths = []
        
        for group_idx, file_group in enumerate(self.sentences):
            for article_idx, article in enumerate(file_group):
                for sentence_idx, sentence in enumerate(article[:-1]):
                    negative_example = sentence 

                    # ensure that it isn't related
                    while negative_example in article: 
                        negative_example = random.choice(random.choice(random.choice(self.sentences)))

                    e = Example(sentence, article[sentence_idx+1], 1) 
                    self.examples.append(e)
                    
                    e = Example(sentence, negative_example, 0)
                    self.examples.append(e)

        print("Precomputing all tokenization in the dataset...")
        for idx, example in tqdm(enumerate(self.examples), total=len(self.examples)):
            context = example.context
            sentence = example.sentence
            encoded_dict = self.tokenizer.encode_plus(text=context, text_pair=sentence, add_special_tokens=True, \
                max_length=self.max_seq_length, truncation_strategy="longest_first", pad_to_max_length=True, \
                return_tensors=None, return_token_type_ids=True, return_attention_mask=True, \
                return_overflowing_tokens=False, return_special_tokens_mask=False) 

            input_ids = encoded_dict['input_ids']
            token_type_ids = encoded_dict['token_type_ids']
            attention_mask = encoded_dict['attention_mask']

            self.memo.append((input_ids, token_type_ids, attention_mask, example.label))


        print(f"{len(self.examples):,} examples created in the dataset.")

    def _precompute_tokenization(self, e):
        idx, example = e
        context = example.context
        sentence = example.sentence
        encoded_dict = self.tokenizer.encode_plus(text=context, text_pair=sentence, add_special_tokens=True, \
          max_length=self.max_seq_length, truncation_strategy="longest_first", pad_to_max_length=True, \
          return_tensors=None, return_token_type_ids=True, return_attention_mask=True, \
          return_overflowing_tokens=False, return_special_tokens_mask=False) 

        input_ids = encoded_dict['input_ids']
        token_type_ids = encoded_dict['token_type_ids']
        attention_mask = encoded_dict['attention_mask']

        return (input_ids, token_type_ids, attention_mask, example.label)

    def __getitem__(self, idx):
        return self.memo[idx] 

    def _add_special_tokens_sentences_pair(self, token_ids_0, token_mask_0, token_ids_1, token_mask_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A RoBERTa sequence pair has the following format: [CLS] A [SEP][SEP] B [SEP]
        """
        sep = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)]
        cls = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)]
        mask = [1] + token_mask_0 + [1] + token_mask_1 + [1]
        input_ids = cls + token_ids_0 + sep + token_ids_1 + sep
        return input_ids, mask

    def __len__(self):
        return len(self.examples) 

    
    def _process_file(self, filename):
        d = None
        lines = []
        with open(filename, "r", encoding="utf-8") as f: 
            lines = f.readlines() 
        sentences = [self._process_line(i) for i in lines]
        return sentences

    def _process_line(self, l):
        d = json.loads(l)
        text = d['text']
        clean = re.compile("<.*?>.*?</.*?>")
        text = re.sub(clean,"", d['text']) 
        sentences = nltk.sent_tokenize(text)
        return sentences

class Example(object):
    def __init__(self,  context, sentence, label):
        self.label = label # 1 means related
        self.context = context 
        self.sentence = sentence

    def __str__(self):
        return f"{self.context} {self.sentence}"
        
if __name__=="__main__":
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    nsp = NextSentenceDataset("out", tokenizer, data_frac=0.10)

    lengths = [] 
    for example in nsp.examples:
        context_tokens = [tokenizer.convert_tokens_to_ids(i) for i in tokenizer.tokenize(example.context)]
        lengths.append(len(context_tokens))
        sentence_tokens = [tokenizer.convert_tokens_to_ids(i) for i in tokenizer.tokenize(example.sentence)]
        lengths.append(len(sentence_tokens))

    print(np.percentile(lengths, 25), np.percentile(lengths, 50), np.percentile(lengths, 75), np.percentile(lengths, 95))
