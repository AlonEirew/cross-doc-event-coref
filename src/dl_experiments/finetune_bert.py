import logging
import random

import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from tqdm import trange
from transformers import BertForSequenceClassification, AdamW, BertTokenizer

from src import LIBRARY_ROOT
from src.utils.dl_utils import get_feat

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def finetune_bert(bert, tokenizer, train, validation, batch_size, epochs=4, use_cuda=True):
    param_optimizer = list(bert.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    for epoch in trange(epochs, desc="Epoch"):
        bert.train()

        tr_loss = 0
        nb_tr_steps = 0

        dataset_size = len(train)
        end_index = batch_size
        random.shuffle(train)
    
        for start_index in trange(0, dataset_size, batch_size, desc="Batches"):
            if end_index > dataset_size:
                end_index = dataset_size

            optimizer.zero_grad()

            batch_features = train[start_index:end_index].copy()
            batch, att_mask, true_label = feat_to_vec(tokenizer, batch_features, use_cuda)
            
            loss = bert(batch, attention_mask=att_mask, labels=true_label)[0]
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            nb_tr_steps += 1
            end_index += batch_size

        bert.eval()
        dev_accuracy = accuracy_on_dataset(bert, tokenizer, validation, use_cuda)
        log('Dev-Acc', epoch, end_index, end_index, tr_loss, 0, dev_accuracy)
        train_accuracy = accuracy_on_dataset(bert, tokenizer, train, use_cuda)
        log('Train-Acc', epoch, end_index, end_index, tr_loss, 0, train_accuracy)


def log(message, epoch, total_count, pair_count, cum_loss, took, accuracy):
    if accuracy != 0.0:
        logger.info('%s: %d: %d: loss: %.10f: Accuracy: %.10f: epoch-took: %dmilli' %
              (message, epoch + 1, total_count, cum_loss / pair_count, accuracy, took))
    else:
        logger.info('%s: %d: %d: loss: %.10f: batch-took: %dmilli' %
              (message, epoch + 1, total_count, cum_loss / pair_count, took))


def feat_to_vec(tokenizer, batch_features, use_cuda):
    MAX_LEN = 128
    MAX_SURROUNDING_CONTX = 10
    ret_golds = list()
    input_ids_list = list()
    att_mask_list = list()
    for mention1, mention2 in batch_features:
        ment1_context = extract_mention_surrounding_context(mention1, MAX_SURROUNDING_CONTX)
        ment2_context = extract_mention_surrounding_context(mention2, MAX_SURROUNDING_CONTX)
        sentence1_words = '[CLS] ' + ' '.join(ment1_context)
        sent1_tokens = tokenizer.tokenize(sentence1_words)
        att_mask = [1] * len(sent1_tokens)

        sentence2_words = ' [SEP]' + ' '.join(ment2_context) + ' [SEP]'
        sent2_tokens = tokenizer.tokenize(sentence2_words)
        att_mask.extend([1] * len(sent2_tokens))

        pair_tokens = sent1_tokens + sent2_tokens
        att_mask.extend([0] * (MAX_LEN - len(pair_tokens)))
        input_ids = tokenizer.convert_tokens_to_ids(pair_tokens)

        if len(input_ids) > MAX_LEN or len(att_mask) > MAX_LEN:
            continue

        gold_label = 1 if mention1.coref_chain == mention2.coref_chain else 0
        ret_golds.append(gold_label)
        input_ids_list.append(input_ids)
        att_mask_list.append(att_mask)

    input_ids_list = pad_sequences(input_ids_list, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    ret_golds = torch.tensor(ret_golds)
    ret_input_ids_list = torch.tensor(input_ids_list)
    ret_att_mask_list = torch.tensor(att_mask_list)

    if use_cuda:
        ret_input_ids_list = ret_input_ids_list.cuda()
        ret_golds = ret_golds.cuda()
        ret_att_mask_list = ret_att_mask_list.cuda()

    return ret_input_ids_list, ret_att_mask_list, ret_golds


def extract_mention_surrounding_context(mention, history_size):
    tokens_inds = mention.tokens_number
    context = mention.mention_context
    start_mention_id = tokens_inds[0]
    end_mention_id = tokens_inds[-1] + 1

    context_before = start_mention_id - history_size
    context_after = end_mention_id + history_size
    if context_before < 0:
        context_before = 0
    if context_after > len(context):
        context_after = len(context)

    return context[context_before:start_mention_id], context[start_mention_id:end_mention_id], \
           context[end_mention_id:context_after]


def accuracy_on_dataset(bert, tokenizer, features, use_cuda):
    dataset_size = len(features)
    batch_size = 50
    end_index = batch_size
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps = 0

    for start_index in range(0, dataset_size, batch_size):
        if end_index > dataset_size:
            end_index = dataset_size

        batch_features = features[start_index:end_index].copy()
        batch, att_mask, true_label = feat_to_vec(tokenizer, batch_features, use_cuda)
        logits = bert(batch, attention_mask=att_mask)
        # Move logits and labels to CPU
        logits = logits[0].detach().cpu().numpy()
        label_ids = true_label.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
        end_index += batch_size

    return eval_accuracy / nb_eval_steps


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def load_datasets(train_file, validation_file, alpha):
    logger.info('Create Features:')
    train_feat = get_feat(train_file, alpha)
    validation_feat = get_feat(validation_file, alpha)
    return train_feat, validation_feat


if __name__ == '__main__':
    # _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Train_Event_gold_mentions.json'
    # _event_validation_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Dev_Event_gold_mentions.json'

    _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Test_Event_gold_mentions.json'
    _event_validation_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Dev_Event_gold_mentions.json'

    _model_out = str(LIBRARY_ROOT) + '/saved_models/wiki_trained_model'

    _learning_rate = 0.01
    _iterations = 4
    _batch_size = 32
    _joint = False
    _type = 'event'
    _alpha = 3
    _use_cuda = False  # args.cuda in ['True', 'true', 'yes', 'Yes']

    _bert = BertForSequenceClassification.from_pretrained('bert-base-cased')
    _tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    if _use_cuda:
        logger.info(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(1)
        _bert.cuda()

    random.seed(1)
    np.random.seed(1)

    _train, _validation = load_datasets(_event_train_file, _event_validation_file, _alpha)
    finetune_bert(_bert, _tokenizer, _train, _validation, _batch_size, _iterations, _use_cuda)
