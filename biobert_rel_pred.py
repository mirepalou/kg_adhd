# -*- coding: utf-8 -*-

'''
Mireia's adaptation of KG-BERT using Bio-BERT
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from transformers import AdamW
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn import metrics
import json


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

def get_features(examples, tokenizer, label_map, task, max_seq_length = 128):
    """
    ALtered kg-bert by mireia to compact code
    """
    features = convert_examples_to_features(examples, max_seq_length, tokenizer, label_map)
    all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f['input_mask'] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f['label_id'] for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if task == 'train':
        sampler = RandomSampler(data) # we break the dependency
    else:
        sampler = SequentialSampler(data)
    
    dataloader = DataLoader(data, sampler=sampler, batch_size=32)

    return dataloader, all_label_ids

def train(model, optimizer, train_dataloader, num_labels):
    '''
    altered from kg-bert
    '''
    model.train()
    for _ in trange(3, desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            # batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, segment_ids, input_mask, labels=None)
            logits = logits.logits ########################################################added by me
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

            # if n_gpu > 1: loss = loss.mean() # mean() to average on multi-gpu.
            # if args.gradient_accumulation_steps > 1: loss = loss / args.gradient_accumulation_steps
            # if args.fp16:  --> Whether to use 16-bit float precision instead of 32-bit optimizer.backward(loss)# else:
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            # if (step + 1) % args.gradient_accumulation_steps == 0:  --> Number of updates steps to accumulate before performing a backward/update pass.
            #     if args.fp16:
            #         # modify learning rate with special warm up BERT uses# if args.fp16 is False, BertAdam is used that handles this automatically
            #         lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step/num_train_optimization_steps,args.warmup_proportion)
            #         for param_group in optimizer.param_groups:param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
        print("Training loss: ", tr_loss, nb_tr_examples)
    # # Save the model (optional)
    # model.save_pretrained('./biobert_triple_classifier')

    return model, tr_loss, nb_tr_steps

def compute_metrics(preds, labels):
        '''
        from kg-bert
        '''
        assert len(preds) == len(labels)
        simple_accuracy = (preds == labels).mean()
        return {"acc": simple_accuracy}

def evaluate(model, dataloader, num_labels, all_label_ids = None, tr_loss=0, nb_tr_steps=0):
    '''
    altered kg-bert to compact
    '''

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc='Evaluating'):

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
            logits = logits.logits

        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    print(preds, preds.shape)
    
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, all_label_ids.numpy())
    loss = tr_loss/nb_tr_steps #if args.do_train else None

    result['eval_loss'] = eval_loss
    result['loss'] = loss
    
    return model, preds, result

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

def triples(data):
  '''
  Takes df and converts into list of triples
  created by Mireia
  '''

  triples = pd.DataFrame()
  triples['entities'] = data['subject_pl'] +' '+ data['property_label'] +' '+ data['object_pl']

  return list(triples['entities'])

def save_predictions(preds, samples, file_name, ground_truth= True):
    '''
    save the predictions in a dataframe with all info
    by Mireia
    '''
    
    scores = ['score_'+str(i) for i in range(preds.shape[1])]
    column_names = ['subject', 'object', 'true_label']+scores
    results = pd.DataFrame(coulmns=column_names)

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

def metrics_predict(preds, test_triples, all_label_ids, label_list, all_triples_str_set, tr_loss, nb_tr_steps, result):
    '''
    compact code + alterations
    '''

    all_label_ids = all_label_ids.numpy()

    ranks = []
    filter_ranks = []
    hits = []
    hits_filter = []
    for i in range(10):
        hits.append([])
        hits_filter.append([])

    for i, pred in enumerate(preds):
        rel_values = torch.tensor(pred)
        _, argsort1 = torch.sort(rel_values, descending=True)
        argsort1 = argsort1.cpu().numpy()
        # print('Sample', i)
        # print('-- True Label:', all_label_ids[i])
        # pred_rel = [(p, np.round(float(rel_values[p]), 3)) for p in argsort1]
        # print('-- Predictions:', pred_rel)

        rank = np.where(argsort1 == all_label_ids[i])[0][0]
        #print(argsort1, all_label_ids[i], rank)
        ranks.append(rank + 1)
        test_triple = test_triples[i]
        filter_rank = rank
        for tmp_label_id in argsort1[:rank]:
            tmp_label = label_list[tmp_label_id]
            tmp_triple = [test_triple[0], tmp_label, test_triple[2]]
            #print(tmp_triple)
            tmp_triple_str = '\t'.join(tmp_triple)
            if tmp_triple_str in all_triples_str_set:
                filter_rank -= 1
        filter_ranks.append(filter_rank + 1)

        for hits_level in range(10):
            if rank <= hits_level:
                hits[hits_level].append(1.0)
            else:
                hits[hits_level].append(0.0)

            if filter_rank <= hits_level:
                hits_filter[hits_level].append(1.0)
            else:
                hits_filter[hits_level].append(0.0)

    print("Raw mean rank: ", np.mean(ranks))
    print("Filtered mean rank: ", np.mean(filter_ranks))
    for i in [0,2,9]:
        print('Raw Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
        print('hits_filter Hits @{0}: {1}'.format(i+1, np.mean(hits_filter[i])))
    preds_t = np.argmax(preds, axis=1)

    result_t = compute_metrics(preds_t, all_label_ids)
    loss = tr_loss/nb_tr_steps #if args.do_train else None

    result_t['eval_loss'] = result['eval_loss']
    result_t['loss'] = loss

    # output_eval_file = os.path.join(args.output_dir, "test_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Test results *****")
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))
    # relation prediction, raw
    print("Relation prediction hits@1, raw...")
    print(metrics.accuracy_score(all_label_ids, preds_t))
    # return 0


def main():
    '''
    Main structure of the code
    '''

    # import data
    # folder_path = "./drive/MyDrive/"
    # data = pd.read_csv("trip2.csv") #folder_path+

    # TRAIN
    # num_train = int(len(data)*0.7) # 70% goes to train
    # train_examples = data[:num_train]
    train_examples = pd.read_csv('train_ad.csv', sep='\t')
    train_labels = train_examples.property_label #label

    labels = set(train_labels) #possible labels
    label_map = {lab: index for index, lab in enumerate(labels)} # mapping integers to relations
    # save mapping
    with open('map_rel.txt', 'w') as file:
        file.write(json.dumps(label_map))

    # VALIDATION
    # num_val = int(len(data)*0.85) # next 15% val and remaining 15% to test
    # eval_examples_0 = data[num_train:num_val]
    eval_examples_0 = pd.read_csv('val_ad.csv', sep='\t')
    eval_examples = eval_examples_0[eval_examples_0['property_label'].isin(labels)].reset_index() # just in case
    print('Validation length:', len(eval_examples_0), len(eval_examples))
    # eval_labels = eval_examples.property_label

    # TEST
    # test_examples_0 = data[num_val:]
    # test_examples = test_examples_0[test_examples_0['property_label'].isin(labels)].reset_index() # just in case
    test_examples = pd.read_csv('test.csv')
    test_labels = test_examples.property_label
    # test_labels = ['interacts with' for i in range(len(test_examples))] #set it as default for formatting

    # Load BioBERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    train_dataloader, all_label_ids_tr = get_features(train_examples, tokenizer, label_map, 'train')
    eval_dataloader, all_label_ids_e = get_features(eval_examples, tokenizer, label_map, 'eval')
    test_dataloader, all_label_ids_tt = get_features(test_examples, tokenizer, label_map, 'test')

    num_labels = len(labels) ##########################################added by me
    model = AutoModelForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1', num_labels = num_labels)
    optimizer = AdamW(model.parameters(), lr=5e-5)  ##### they use others
    
    # 1 training
    print('Training')
    model, tr_loss, nb_tr_steps = train(model, optimizer, train_dataloader, num_labels)

    # 2 validation
    print('Validation')
    model, preds_val, result = evaluate(model, eval_dataloader, num_labels, all_label_ids_e, tr_loss, nb_tr_steps)
    save_predictions(preds_val, eval_examples, 'val_preds.csv')

    # 3 test
    # train_triples = triples(train_examples)
    # dev_triples = triples(eval_examples)
    # test_triples = triples(test_examples)
    # all_triples = train_triples + dev_triples + test_triples
    # all_triples_str_set = set(all_triples)

    # preds = evaluate(model, test_dataloader, num_labels, 'Testing')
    print('Testing')
    preds = predict(model, test_dataloader)
    save_predictions(preds, test_examples, 'test_preds.csv')

    ###### still need to check what is exactly going on
    # metrics_predict(preds, test_triples, all_label_ids_tt, list(test_labels), all_triples_str_set, tr_loss, nb_tr_steps, result)





if __name__ == "__main__":
    main()