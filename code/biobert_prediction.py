# -*- coding: utf-8 -*-

'''
Mireia's adaptation of KG-BERT using Bio-BERT
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn import metrics
import json
import os


### Embedding data

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """
    Truncates a sequence pair in place to the maximum length.
    Retrieved from KG-BERT
    """

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, max_seq_length, tokenizer, label_map):
    """
    Loads a data file into a list of `InputBatch`s.
    Retrieved from KG-BERT but slightly altered to fit the data
    """

    features = []

    text_a = examples['subject_pl']
    text_b = examples['object_pl']
    label = examples['property_label']

    for ex_index in range(len(examples)):
        # if ex_index % 10000 == 0 and print_info:
            # logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(text_a[ex_index])

        tokens_b = None
        if text_b[ex_index]:
            tokens_b = tokenizer.tokenize(text_b[ex_index])
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[label[ex_index]]

        features.append({'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, 'label_id': label_id})
    return features

def get_features(examples, tokenizer, label_map, max_seq_length = 128):
    """
    ALtered kg-bert by mireia to compact code
    """
    features = convert_examples_to_features(examples, max_seq_length, tokenizer, label_map)
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f['label_id'] for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    sampler = SequentialSampler(data)
    
    dataloader = DataLoader(data, sampler=sampler, batch_size=32)

    return dataloader, all_label_ids



def predict(model, dataloader):
    # model.eval()
    nb_test_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc='Testing'):

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
            logits = logits.logits

        nb_test_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    print(preds, preds.shape)
    return preds



def save_predictions(preds, samples, file_name, ground_truth= True):
    '''
    save the predictions in a dataframe with all info
    by Mireia
    '''
    
    scores = ['score_'+str(i) for i in range(preds.shape[1])]
    column_names = ['subject', 'object', 'true_label']+scores
    results = pd.DataFrame(columns=column_names)

    for i, pred in enumerate(preds):
        # get the scores for each possible label
        rel_values = torch.tensor(pred)
        _, argsort1 = torch.sort(rel_values, descending=True) # sort predictions
        argsort1 = argsort1.cpu().numpy()
        pred_rel = {'score_'+str(p): np.round(float(rel_values[p]), 3) for p in argsort1}

        sample = samples.loc[i]
        tl = None
        if ground_truth:
            tl = sample['property_label']
        res = {'subject': sample['subject_pl'], 'object': sample['object_pl'], 'true_label': tl}
        res.update(pred_rel)

        results.loc[i] = pd.Series(res)

    
    results.to_csv(file_name)


def main():
    '''
    Main structure of the code
    '''

    # TEST
    test_examples = pd.read_csv('test_iron_reduced.csv')
    test_labels = test_examples.property_label
    
    ##########################################added by me
    label_map = json.load(open('map_rel_bs8.txt'))
    num_labels = len(label_map) 

    # Load a trained model and vocabulary that you have fine-tuned
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(script_dir, 'saved_model_bs8')
    model = AutoModelForSequenceClassification.from_pretrained(save_directory, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(save_directory)

    test_dataloader, all_label_ids_tt = get_features(test_examples, tokenizer, label_map)

    
    # 3 test    
    print('Testing')
    preds = predict(model, test_dataloader)
    save_predictions(preds, test_examples, 'test_preds_iron_final.csv', False)




if __name__ == "__main__":
    main()
