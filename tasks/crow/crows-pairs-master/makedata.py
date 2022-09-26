import os
import csv
import json
import jsonlines
import torch
import argparse
import difflib
import logging
import numpy as np
import pandas as pd
from icetk import icetk
def get_span(seq1, seq2):
    """
    This function extract spans that are shared between two sequences.
    """

    seq1 = [str(x) for x in seq1]
    seq2 = [str(x) for x in seq2]

    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    template1, template2 = [], []
    diff1, diff2 = [], []
    for op in matcher.get_opcodes():
        # each op is a list of tuple: 
        # (operation, pro_idx_start, pro_idx_end, anti_idx_start, anti_idx_end)
        # possible operation: replace, insert, equal
        # https://docs.python.org/3/library/difflib.html
        if op[0] == 'equal':
            template1 += [x for x in range(op[1], op[2], 1)]
            template2 += [x for x in range(op[3], op[4], 1)]
        else:
            diff1 += [x for x in range(op[1], op[2], 1)]
            diff2 += [x for x in range(op[3], op[4], 1)]

    return template1, template2, diff1, diff2

def read_data(input_file):
    """
    Load data into pandas DataFrame format.
    """

    df_data = pd.DataFrame(columns=['sent1', 'sent2', 'direction', 'bias_type'])
    MASK_ID = 150000

    x = 0
    with jsonlines.open("data/CROWS/CROWS_dataset.jsonl", 'w') as w:
        with open(input_file) as f:       
            reader = csv.DictReader(f)
            for row in reader:
                
                direction, gold_bias = '_', '_'
                direction = row['stereo_antistereo']
                bias_type = row['bias_type']

                sent1, sent2 = '', ''
                if direction == 'stereo':
                    sent1 = row['sent_more']
                    sent2 = row['sent_less']
                else:
                    sent1 = row['sent_less']
                    sent2 = row['sent_more']

                encode_1 = icetk.encode(sent1)
                encode_2 = icetk.encode(sent2)
                template1, template2, diff1, diff2 = get_span(encode_1,encode_2)
                assert len(template1) == len(template2)
                for pos in template1:
                    tem_encode = encode_1.copy()
                    choices = [icetk.decode([tem_encode[pos]])]
                    if choices[0] == "":
                        print(0)
                        continue
                    tem_encode[pos] = MASK_ID
                    w.write({"inputs":tem_encode,"choices_pretokenized":choices,"label":0,"bias_type":bias_type,"pair_ID":x,"sent_ID":1})
                for pos in template2:
                    tem_encode = encode_2.copy()
                    choices = [icetk.decode([tem_encode[pos]])]
                    if choices[0] == "":
                        print(0)
                        continue
                    tem_encode[pos] = MASK_ID
                    w.write({"inputs":tem_encode,"choices_pretokenized":choices,"label":0,"bias_type":bias_type,"pair_ID":x,"sent_ID":2})
                    
                x = x + 1
               
'''
def read_data(input_file):
    """
    Load data into pandas DataFrame format.
    """
    # 生成MASK了不同位置的
    df_data = pd.DataFrame(columns=['sent1', 'sent2', 'direction', 'bias_type'])
    MASK_ID = 150000

    x = 0
    with jsonlines.open("data/CROWS/test2_10.jsonl", 'w') as w:
        with open(input_file) as f:       
            reader = csv.DictReader(f)
            for row in reader:
                
                direction, gold_bias = '_', '_'
                direction = row['stereo_antistereo']
                bias_type = row['bias_type']

                sent1, sent2 = '', ''
                if direction == 'stereo':
                    sent1 = row['sent_more']
                    sent2 = row['sent_less']
                else:
                    sent1 = row['sent_less']
                    sent2 = row['sent_more']

                encode_1 = icetk.encode(sent1)
                encode_2 = icetk.encode(sent2)
                template1, template2, diff1, diff2 = get_span(encode_1,encode_2)
                assert len(template1) == len(template2)

                if len(diff1)>1 or len(diff1) != len(diff2):
                    continue
                for code in diff1:
                    #choices = [icetk.decode([encode_1[code]]),icetk.decode([encode_2[code]])]
                    choices = [icetk.decode([encode_1[code]])]
                    encode_1[code] = MASK_ID
                    encode_2[code] = MASK_ID
                    
                

                df_item = {'sent1': sent1,
                       'sent2': sent2,
                       'direction': direction,
                       'bias_type': bias_type}
                df_data = df_data.append(df_item, ignore_index=True)
                x = x + 1
                if x==10:
                    break
                w.write({"inputs":encode_1,"choices_pretokenized":choices,"label":0})
    return df_data'''

read_data("tasks/crow/crows-pairs-master/data/crows_pairs_anonymized.csv")
