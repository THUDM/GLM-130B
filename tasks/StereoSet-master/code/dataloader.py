import json
import string
from tqdm import tqdm

class SentimentIntrasentenceLoader(object):
    def __init__(self, tokenizer, max_seq_length=None, pad_to_max_length=False, input_file="../../data/bias.json"):
        stereoset = StereoSet(input_file)
        clusters = stereoset.get_intrasentence_examples()
        self.tokenizer = tokenizer
        self.sentences = []
        self.MASK_TOKEN = self.tokenizer.mask_token
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length

        if tokenizer.__class__.__name__=="XLNetTokenizer":
            self.prepend_text = """In 1991, the remains of Russian Tsar Nicholas II and his family
            (except for Alexei and Maria) are discovered.
            The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
            remainder of the story. 1883 Western Siberia,
            a young Grigori Rasputin is asked by his father and a group of men to perform magic.
            Rasputin has a vision and denounces one of the men as a horse thief. Although his
            father initially slaps him for making such an accusation, Rasputin watches as the
            man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
            the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
            with people, even a bishop, begging for his blessing. <eod> """

        for cluster in clusters:
            for sentence in cluster.sentences:
                new_sentence = cluster.context.replace("BLANK", sentence.template_word)
                self.sentences.append((new_sentence, sentence.ID))

    def __len__(self):
        return len(self.sentences)  

    def __getitem__(self, idx):
        sentence, sentence_id = self.sentences[idx]
        if self.tokenizer.__class__.__name__=="XLNetTokenizer":
            text = self.prepend_text
            text_pair = sentence
        else:
            text = sentence
            text_pair = None
        tokens_dict = self.tokenizer.encode_plus(text, text_pair=text_pair, add_special_tokens=True, max_length=self.max_seq_length, \
            pad_to_max_length=self.pad_to_max_length, return_token_type_ids=True, return_attention_mask=True, \
            return_overflowing_tokens=False, return_special_tokens_mask=False, return_tensors="pt")
        input_ids = tokens_dict['input_ids']
        attention_mask = tokens_dict['attention_mask']
        token_type_ids = tokens_dict['token_type_ids']
        return sentence_id, input_ids, attention_mask, token_type_ids 

class IntrasentenceLoader(object):
    def __init__(self, tokenizer, max_seq_length=None, pad_to_max_length=False, input_file="../../data/bias.json"):
        stereoset = StereoSet(input_file)
        clusters = stereoset.get_intrasentence_examples()
        self.tokenizer = tokenizer
        self.sentences = []
        self.MASK_TOKEN = self.tokenizer.mask_token
        self.max_seq_length = max_seq_length
        self.pad_to_max_length = pad_to_max_length

        if tokenizer.__class__.__name__=="XLNetTokenizer":
            self.prepend_text = """In 1991, the remains of Russian Tsar Nicholas II and his family
            (except for Alexei and Maria) are discovered.
            The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
            remainder of the story. 1883 Western Siberia,
            a young Grigori Rasputin is asked by his father and a group of men to perform magic.
            Rasputin has a vision and denounces one of the men as a horse thief. Although his
            father initially slaps him for making such an accusation, Rasputin watches as the
            man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
            the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
            with people, even a bishop, begging for his blessing. <eod> """

        for cluster in clusters:
            for sentence in cluster.sentences:
                insertion_tokens = self.tokenizer.encode(sentence.template_word, add_special_tokens=False)
                for idx in range(len(insertion_tokens)):
                    insertion = self.tokenizer.decode(insertion_tokens[:idx])
                    insertion_string = f"{insertion}{self.MASK_TOKEN}"
                    new_sentence = cluster.context.replace("BLANK", insertion_string)
                    # print(new_sentence, self.tokenizer.decode([insertion_tokens[idx]]))
                    next_token = insertion_tokens[idx]
                    self.sentences.append((new_sentence, sentence.ID, next_token))

    def __len__(self):
        return len(self.sentences)  

    def __getitem__(self, idx):
        sentence, sentence_id, next_token = self.sentences[idx]
        if self.tokenizer.__class__.__name__=="XLNetTokenizer":
            text = self.prepend_text
            text_pair = sentence
        else:
            text = sentence
            text_pair = None
        tokens_dict = self.tokenizer.encode_plus(text, text_pair=text_pair, add_special_tokens=True, max_length=self.max_seq_length, \
            pad_to_max_length=self.pad_to_max_length, return_token_type_ids=True, return_attention_mask=True, \
            return_overflowing_tokens=False, return_special_tokens_mask=False)
        input_ids = tokens_dict['input_ids']
        attention_mask = tokens_dict['attention_mask']
        token_type_ids = tokens_dict['token_type_ids']
        return sentence_id, next_token, input_ids, attention_mask, token_type_ids 
         
class StereoSet(object):
    def __init__(self, location, json_obj=None):
        """
        Instantiates the StereoSet object.

        Parameters
        ----------
        location (string): location of the StereoSet.json file.
        """

        if json_obj==None:
            with open(location, "r") as f:
                self.json = json.load(f)
        else:
            self.json = json_obj

        self.version = self.json['version']
        self.intrasentence_examples = self.__create_intrasentence_examples__(
            self.json['data']['intrasentence'])
        self.intersentence_examples = self.__create_intersentence_examples__(
            self.json['data']['intersentence'])

    def __create_intrasentence_examples__(self, examples):
        created_examples = []
        for example in examples:
            sentences = []
            for sentence in example['sentences']:
                labels = []
                for label in sentence['labels']:
                    labels.append(Label(**label))
                sentence_obj = Sentence(
                    sentence['id'], sentence['sentence'], labels, sentence['gold_label'])
                word_idx = None
                for idx, word in enumerate(example['context'].split(" ")):
                    if "BLANK" in word: 
                        word_idx = idx
                if word_idx is None:
                    raise Exception("No blank word found.")
                template_word = sentence['sentence'].split(" ")[word_idx]
                sentence_obj.template_word = template_word.translate(str.maketrans('', '', string.punctuation))
                sentences.append(sentence_obj)
            created_example = IntrasentenceExample(
                example['id'], example['bias_type'], 
                example['target'], example['context'], sentences) 
            created_examples.append(created_example)
        return created_examples

    def __create_intersentence_examples__(self, examples):
        created_examples = []
        for example in examples:
            sentences = []
            for sentence in example['sentences']:
                labels = []
                for label in sentence['labels']:
                    labels.append(Label(**label))
                sentence = Sentence(
                    sentence['id'], sentence['sentence'], labels, sentence['gold_label'])
                sentences.append(sentence)
            created_example = IntersentenceExample(
                example['id'], example['bias_type'], example['target'], 
                example['context'], sentences) 
            created_examples.append(created_example)
        return created_examples
    
    def get_intrasentence_examples(self):
        return self.intrasentence_examples

    def get_intersentence_examples(self):
        return self.intersentence_examples

class Example(object):
    def __init__(self, ID, bias_type, target, context, sentences):
        """
         A generic example.

         Parameters
         ----------
         ID (string): Provides a unique ID for the example.
         bias_type (string): Provides a description of the type of bias that is 
             represented. It must be one of [RACE, RELIGION, GENDER, PROFESSION]. 
         target (string): Provides the word that is being stereotyped.
         context (string): Provides the context sentence, if exists,  that 
             sets up the stereotype. 
         sentences (list): a list of sentences that relate to the target. 
         """

        self.ID = ID
        self.bias_type = bias_type
        self.target = target
        self.context = context
        self.sentences = sentences

    def __str__(self):
        s = f"Domain: {self.bias_type} - Target: {self.target} \r\n"
        s += f"Context: {self.context} \r\n" 
        for sentence in self.sentences:
            s += f"{sentence} \r\n" 
        return s

class Sentence(object):
    def __init__(self, ID, sentence, labels, gold_label):
        """
        A generic sentence type that represents a sentence.

        Parameters
        ----------
        ID (string): Provides a unique ID for the sentence with respect to the example.
        sentence (string): The textual sentence.
        labels (list of Label objects): A list of human labels for the sentence. 
        gold_label (enum): The gold label associated with this sentence, 
            calculated by the argmax of the labels. This must be one of 
            [stereotype, anti-stereotype, unrelated, related].
        """

        assert type(ID)==str
        assert gold_label in ['stereotype', 'anti-stereotype', 'unrelated']
        assert isinstance(labels, list)
        assert isinstance(labels[0], Label)

        self.ID = ID
        self.sentence = sentence
        self.gold_label = gold_label
        self.labels = labels
        self.template_word = None

    def __str__(self):
        return f"{self.gold_label.capitalize()} Sentence: {self.sentence}"

class Label(object):
    def __init__(self, human_id, label):
        """
        Label, represents a label object for a particular sentence.

        Parameters
        ----------
        human_id (string): provides a unique ID for the human that labeled the sentence.
        label (enum): provides a label for the sentence. This must be one of 
            [stereotype, anti-stereotype, unrelated, related].
        """
        assert label in ['stereotype',
                         'anti-stereotype', 'unrelated', 'related']
        self.human_id = human_id
        self.label = label


class IntrasentenceExample(Example):
    def __init__(self, ID, bias_type, target, context, sentences):
        """
        Implements the Example class for an intrasentence example.

        See Example's docstring for more information.
        """
        super(IntrasentenceExample, self).__init__(
            ID, bias_type, target, context, sentences)


class IntersentenceExample(Example):
    def __init__(self, ID, bias_type, target, context, sentences):
        """
        Implements the Example class for an intersentence example.

        See Example's docstring for more information.
        """
        super(IntersentenceExample, self).__init__(
            ID, bias_type, target, context, sentences)
