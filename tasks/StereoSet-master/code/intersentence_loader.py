from os import path 
import sys
sys.path.append("..")
import dataloader
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import LabelEncoder

class IntersentenceDataset(Dataset):
    def __init__(self, tokenizer, args): 
        self.tokenizer = tokenizer
        filename = args.input_file
        dataset = dataloader.StereoSet(filename)
        self.emp_max_seq_length = float("-inf")
        self.max_seq_length = args.max_seq_length
        self.batch_size = args.batch_size

        if self.tokenizer.__class__.__name__=="XLNetTokenizer":
            self.prepend_text = """ In 1991, the remains of Russian Tsar Nicholas II and his family
		(except for Alexei and Maria) are discovered.
		The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
		remainder of the story. 1883 Western Siberia,
		a young Grigori Rasputin is asked by his father and a group of men to perform magic.
		Rasputin has a vision and denounces one of the men as a horse thief. Although his
		father initially slaps him for making such an accusation, Rasputin watches as the
		man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
		the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
		with people, even a bishop, begging for his blessing. <eod> </s> <eos> """ 
            self.prepend_text = None
        else:
            self.prepend_text = None

        intersentence_examples = dataset.get_intersentence_examples()
        
        self.preprocessed = [] 
        for example in intersentence_examples:
            context = example.context
            if self.prepend_text is not None:
                context = self.prepend_text + context 
            for sentence in example.sentences:
                # if self.tokenizer.__class__.__name__ in ["XLNetTokenizer", "RobertaTokenizer"]:
                if self.tokenizer.__class__.__name__ in ["XLNetTokenizer", "RobertaTokenizer", "BertTokenizer"]:
                    # support legacy pretrained NSP heads!
                    input_ids, token_type_ids = self._tokenize(context, sentence.sentence)
                    attention_mask = [1 for _ in input_ids] 
                    self.preprocessed.append((input_ids, token_type_ids, attention_mask, sentence.ID))  
                else:
                    encoded_dict = self.tokenizer.encode_plus(text=context, text_pair=sentence.sentence, add_special_tokens=True, max_length=self.max_seq_length, truncation_strategy="longest_first", pad_to_max_length=False, return_tensors=None, return_token_type_ids=True, return_attention_mask=True, return_overflowing_tokens=False, return_special_tokens_mask=False) 
                    # prior tokenization
                    # input_ids, position_ids, attention_mask = self._tokenize(context, sentence)

                    input_ids = encoded_dict['input_ids']
                    token_type_ids = encoded_dict['token_type_ids']
                    attention_mask = encoded_dict['attention_mask']
                    self.preprocessed.append((input_ids, token_type_ids, attention_mask, sentence.ID))

        print(f"Maximum sequence length found: {self.emp_max_seq_length}")
         
    def __len__(self):
        return len(self.preprocessed) 

    def __getitem__(self, idx):
        input_ids, token_type_ids, attention_mask, sentence_id = self.preprocessed[idx]
        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        return input_ids, token_type_ids, attention_mask, sentence_id 

    def _tokenize(self, context, sentence):
        # context = "Q: " + context
        context_tokens = self.tokenizer.tokenize(context)
        context_tokens = [self.tokenizer.convert_tokens_to_ids(i) for i in context_tokens]

        # sentence = "A: " + sentence
        sentence_tokens = self.tokenizer.tokenize(sentence)
        if self.batch_size>1:
            if (len(sentence_tokens) + len(context_tokens)) > self.emp_max_seq_length:
                self.emp_max_seq_length = (len(sentence_tokens) + len(context_tokens))
            while (len(sentence_tokens) + len(context_tokens)) < self.max_seq_length:
                sentence_tokens.append(self.tokenizer.pad_token)
        sentence_tokens = [self.tokenizer.convert_tokens_to_ids(i) for i in sentence_tokens] 

        input_ids = self.add_special_tokens_sequence_pair(context_tokens, sentence_tokens) 
        if self.batch_size>1:
            input_ids = input_ids[:self.max_seq_length]
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        
        # get the position ids
        position_offset = input_ids.index(sep_token_id) 
        assert position_offset>0
        position_ids = [1 if idx>position_offset else 0 for idx in range(len(input_ids))] 
        return input_ids, position_ids

    def add_special_tokens_sequence_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A RoBERTa sequence pair has the following format: <s> A </s></s> B </s>
        """
        sep = [self.tokenizer.sep_token_id]
        cls = [self.tokenizer.cls_token_id]
        if self.tokenizer.__class__.__name__=="XLNetTokenizer":
            return token_ids_0 + sep + token_ids_1 + sep + cls
        elif self.tokenizer.__class__.__name__=="RobertaTokenizer":
            return cls + token_ids_0 + sep + sep + token_ids_1 + sep
        elif self.tokenizer.__class__.__name__=="BertTokenizer":
            return cls + token_ids_0 + sep + token_ids_1 + sep

class SentimentIntersentenceDataset(Dataset):
    def __init__(self, tokenizer, args): 
        self.tokenizer = tokenizer
        filename = args.input_file
        dataset = dataloader.StereoSet(filename)
        self.emp_max_seq_length = float("-inf")
        self.max_seq_length = args.max_seq_length
        self.batch_size = args.batch_size

        if self.tokenizer.__class__.__name__=="XLNetTokenizer":
            self.prepend_text = """ In 1991, the remains of Russian Tsar Nicholas II and his family
		(except for Alexei and Maria) are discovered.
		The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
		remainder of the story. 1883 Western Siberia,
		a young Grigori Rasputin is asked by his father and a group of men to perform magic.
		Rasputin has a vision and denounces one of the men as a horse thief. Although his
		father initially slaps him for making such an accusation, Rasputin watches as the
		man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
		the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
		with people, even a bishop, begging for his blessing. <eod> </s> <eos> """ 
            self.prepend_text = None
        else:
            self.prepend_text = None

        intersentence_examples = dataset.get_intersentence_examples()

        self.preprocessed = [] 
        for example in intersentence_examples:
            context = example.context
            if self.prepend_text is not None:
                context = self.prepend_text + context 
            for sentence in example.sentences:
                # if self.tokenizer.__class__.__name__ in ["XLNetTokenizer", "RobertaTokenizer"]:
                if self.tokenizer.__class__.__name__ in ["XLNetTokenizer", "RobertaTokenizer"]: #, "BertTokenizer"]:
                    # support legacy pretrained NSP heads!
                    input_ids, token_type_ids = self._tokenize(context, sentence.sentence)
                    attention_mask = [1 for _ in input_ids] 
                    self.preprocessed.append((input_ids, token_type_ids, attention_mask, sentence.ID))  
                else:
                    s = f"{context} {sentence.sentence}"
                    pad_to_max_length = self.batch_size>1
                    encoded_dict = self.tokenizer.encode_plus(text=context, text_pair=sentence.sentence, add_special_tokens=True, max_length=self.max_seq_length, truncation_strategy="longest_first", pad_to_max_length=pad_to_max_length, return_tensors="pt", return_token_type_ids=True, return_attention_mask=True, return_overflowing_tokens=False, return_special_tokens_mask=False) 
                    # prior tokenization
                    # input_ids, position_ids, attention_mask = self._tokenize(context, sentence)

                    input_ids = encoded_dict['input_ids']
                    token_type_ids = encoded_dict['token_type_ids']
                    attention_mask = encoded_dict['attention_mask']
                    self.preprocessed.append((input_ids, token_type_ids, attention_mask, sentence.ID))

        print(f"Maximum sequence length found: {self.emp_max_seq_length}")
         
    def __len__(self):
        return len(self.preprocessed) 

    def __getitem__(self, idx):
        input_ids, token_type_ids, attention_mask, sentence_id = self.preprocessed[idx]
        # input_ids = torch.tensor(input_ids)
        # token_type_ids = torch.tensor(token_type_ids)
        # attention_mask = torch.tensor(attention_mask)
        return sentence_id, input_ids, attention_mask, token_type_ids 

    def _tokenize(self, context, sentence):
        # context = "Q: " + context
        context_tokens = self.tokenizer.tokenize(context)
        context_tokens = [self.tokenizer.convert_tokens_to_ids(i) for i in context_tokens]

        # sentence = "A: " + sentence
        sentence_tokens = self.tokenizer.tokenize(sentence)
        if self.batch_size>1:
            if (len(sentence_tokens) + len(context_tokens)) > self.emp_max_seq_length:
                self.emp_max_seq_length = (len(sentence_tokens) + len(context_tokens))
            while (len(sentence_tokens) + len(context_tokens)) < self.max_seq_length:
                sentence_tokens.append(self.tokenizer.pad_token)
        sentence_tokens = [self.tokenizer.convert_tokens_to_ids(i) for i in sentence_tokens] 

        input_ids = self.add_special_tokens_sequence_pair(context_tokens, sentence_tokens) 
        if self.batch_size>1:
            input_ids = input_ids[:self.max_seq_length]
        sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        
        # get the position ids
        position_offset = input_ids.index(sep_token_id) 
        assert position_offset>0
        position_ids = [1 if idx>position_offset else 0 for idx in range(len(input_ids))] 
        return input_ids, position_ids

    def add_special_tokens_sequence_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        A RoBERTa sequence pair has the following format: <s> A </s></s> B </s>
        """
        sep = [self.tokenizer.sep_token_id]
        cls = [self.tokenizer.cls_token_id]
        if self.tokenizer.__class__.__name__=="XLNetTokenizer":
            return token_ids_0 + sep + token_ids_1 + sep + cls
        elif self.tokenizer.__class__.__name__=="RobertaTokenizer":
            return cls + token_ids_0 + sep + sep + token_ids_1 + sep
        elif self.tokenizer.__class__.__name__=="BertTokenizer":
            return cls + token_ids_0 + sep + token_ids_1 + sep

