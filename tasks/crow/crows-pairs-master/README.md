# CrowS-Pairs

This is the Github repo for [CrowS-Pairs](https://www.aclweb.org/anthology/2020.emnlp-main.154/), a challenge dataset for measuring the degree to which U.S. stereotypical biases present in the masked language models (MLMs). The associated paper is to be published as part of The 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020).

**Data reliability: Please note that recent work by [Blodgett et al. (2021)](https://www.microsoft.com/en-us/research/uploads/prod/2021/06/The_Salmon_paper.pdf) has found significant issues with noise and reliability of the data in CrowS-Pairs. The problems are significant enough that CrowS-Pairs may not be a good indicator of the presence of social biases in LMs. Please refer to the Blodgett et al. paper for details.**

## The Dataset

The dataset along with its annotations is in [crows_pairs_anonymized.csv](https://github.com/nyu-mll/crows-pairs/blob/master/data/crows_pairs_anonymized.csv). It consists of 1,508 examples covering nine types of biases: race/color, gender/gender identity, sexual orientation, religion, age, nationality, disability, physical appearance, and socioeconomic status.

Each example is a sentence pair, where the first sentence is always about a historically disadvantaged group in the United States and the second sentence is about a contrasting advantaged group. The first sentence can _demonstrate_ or _violate_ a stereotype. The other sentence is a minimal edit of the first sentence: The only words that change between them are those that identify the group. Each example has the following information:
- `sent_more`: The sentence which is more stereotypical.
- `sent_less`: The sentence which is less stereotypical.
- `stereo_antistereo`: The stereotypical direction of the pair. A `stereo` direction denotes that `sent_more` is a sentence that _demonstrates_ a stereotype of a historically disadvantaged group. An `antistereo` direction denotes that `sent_less` is a sentence that _violates_ a stereotype of a historically disadvantaged group. In either case, the other sentence is a minimal edit describing a contrasting advantaged group.
- `bias_type`: The type of biases present in the example.
- `annotations`: The annotations of bias types from crowdworkers.
- `anon_writer`: The _anonymized_ id of the writer.
- `anon_annotators`: The _anonymized_ ids of the annotators.

## Quantifying Stereotypical Biases in MLMs

### Requirement

Use Python 3 (we use Python 3.7) and install the required packages.

```
pip install -r requirements.txt
```

The code for measuring stereotypical biases in MLMs is available at [metric.py](https://github.com/nyu-mll/crows-pairs/blob/master/metric.py). You can run the code using the following command:
```
python metric.py 
	--input_file data/crows_pairs_anonymized.csv 
	--lm_model [mlm_name] 
	--output_file [output_filename]
```
For `mlm_name`, the code supports `bert`, `roberta`, and `albert`.

The `--output_file` will store the sentence scores for each example. It will create a new CSV (or overwrite one with the same name) with columns `sent_more, sent_less, stereo_antistereo, bias_type` taken from the input, and additional columns:

- `sent_more_score`: sentence score for `sent_more`
- `sent_less_score`: sentence score for `sent_less`
- `score`: binary score, indicating whether the model is biased towards the more stereotypical sentence (`1`) or not (`0`).

Please refer to the paper on how we calculate the sentence score.

Note that, if you use a newer version of transformers (3.x.x), you might obtain different scores than the one reported in our paper.

## License

CrowS-Pairs is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/). It is created using prompts taken from the [ROCStories corpora](https://cs.rochester.edu/nlp/rocstories/) and the fiction part of [MNLI](https://cims.nyu.edu/~sbowman/multinli/). Please refer to their papers for more details.

## Reference

If you use CrowS-Pairs or our metric in your work, please cite it directly:

```
@inproceedings{nangia2020crows,
    title = "{CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models}",
    author = "Nangia, Nikita  and
      Vania, Clara  and
      Bhalerao, Rasika  and
      Bowman, Samuel R.",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics"
}
```




