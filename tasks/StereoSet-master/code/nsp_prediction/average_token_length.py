import dataloader
import pytorch_transformers
import os
import dataset
from scipy import stats

tokenizer = getattr(pytorch_transformers, "GPT2Tokenizer").from_pretrained("gpt2")

filename = os.path.join(os.path.abspath(
            __file__ + "/../../.."), "data/bias.json")

data = dataset.NextSentenceDataset("out", tokenizer, data_frac=args.data_frac, max_seq_length=args.max_seq_length, test=args.test)

intersentence_examples = dataset.get_intersentence_examples()

lengths = []
for cluster in intersentence_examples:
    for sentence in cluster.sentences:
        s = f"{cluster.context} {sentence.sentence}" 
        print(s)
        lengths.append(len(tokenizer.tokenize(s)))

print(f"Average token length: {stats.describe(lengths)}")
