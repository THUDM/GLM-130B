# Next Sentence Prediction (NSP)
This folder contains code for training a next sentence prediction head to evaluate bias on intersentence tasks. We use Wikipedia dumps (as suggested by Devlin et al.) to train the next sentence prediction head.


## Obtaining a Wikipedia Dump
Download the [latest dump of Wikipedia](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), and extract the text with `[WikiExtractor.py](https://github.com/attardi/wikiextractor)`, and pass the path to the `--dataset` argument in `main.py`.

## Compute Requirements
On average, we used 3-4 2080 Ti's to fine-tune the models. For GPT-2, we recommend the use of [Apex](https://nvidia.github.io/apex/amp.html) to utilize FP16 to fit GPT2-medium in 12GB of RAM. For GPT-2 large, we used 4 Tesla V100s.

## Pretrained Models
For reproducibility, we release our pretrained models to the general public. 
