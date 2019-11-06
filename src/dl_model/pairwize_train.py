import logging
import random

import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from src import LIBRARY_ROOT
from src.dl_experiments.finetune_bert import load_datasets, extract_mention_surrounding_context
from src.dl_model.pairwize_model import PairWiseModel

logger = logging.getLogger(__name__)


def train_pairwise(bert, tokenizer, pairwize_model, train, validation, batch_size, epochs=4, lr=2e-5, use_cuda=True):

    for epoch in range(epochs): #, desc="Epoch"
        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(pairwize_model.parameters(), lr=2e-5)

        dataset_size = len(train)
        end_index = batch_size
        random.shuffle(train)

        cum_loss = 0.0
        for start_index in range(0, dataset_size, batch_size): #, desc="Batches"
            if end_index > dataset_size:
                end_index = dataset_size

            optimizer.zero_grad()

            batch_features = train[start_index:end_index].copy()
            embeded_features, gold_labels = get_bert_rep(bert, tokenizer, batch_features, use_cuda)
            output = pairwize_model(embeded_features)

            loss = loss_func(output, gold_labels)
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            end_index += batch_size

            logger.info('%d: %d: loss: %.10f:' % (epoch + 1, end_index, cum_loss / end_index))

        dev_accuracy = accuracy_on_dataset(bert, tokenizer, pairwize_model, validation, use_cuda)
        logger.info('%s: %d: %d: loss: %.10f: Accuracy: %.10f' %
                    ('Dev-Acc', epoch + 1, end_index, cum_loss / end_index, dev_accuracy))
        # train_accuracy = accuracy_on_dataset(bert, tokenizer, train, use_cuda)
        # log('Train-Acc', epoch, end_index, end_index, tr_loss, 0, train_accuracy)


def accuracy_on_dataset(bert, tokenizer, pairwize_model, features, use_cuda):
    dataset_size = len(features)
    batch_size = 1000
    end_index = batch_size
    labels = list()
    predictions = list()
    for start_index in range(0, dataset_size, batch_size):
        if end_index > dataset_size:
            end_index = dataset_size

        batch_features = features[start_index:end_index].copy()
        batch, batch_label = get_bert_rep(bert, tokenizer, batch_features, use_cuda)
        batch_predictions = pairwize_model.predict(batch)

        predictions.append(batch_predictions)
        labels.append(batch_label)
        end_index += batch_size

    return torch.mean((torch.cat(labels) == torch.cat(predictions)).float())


def get_bert_rep(bert, tokenizer, batch_features, use_cuda):
    MAX_SURROUNDING_CONTX = 10
    batch_result = list()
    batch_labels = list()
    for mention1, mention2 in batch_features:
        ment1_ids, ment1_inx_start, ment1_inx_end = mention_feat_to_vec(tokenizer, mention1, MAX_SURROUNDING_CONTX)
        ment2_ids, ment2_inx_start, ment2_inx_end = mention_feat_to_vec(tokenizer, mention2, MAX_SURROUNDING_CONTX)

        with torch.no_grad():
            if use_cuda:
                ment1_ids = ment1_ids.cuda()
                ment2_ids = ment2_ids.cuda()

            all_hidden_states1, _ = bert(ment1_ids)[-2:]
            all_hidden_states2, _ = bert(ment2_ids)[-2:]

        # last_attend2 = all_attentions2[0]
        last_hidden1_span = all_hidden_states1[0].view(all_hidden_states1[0].shape[1], -1)[ment1_inx_start:ment1_inx_end]
        last_hidden2_span = all_hidden_states2[0].view(all_hidden_states2[0].shape[1], -1)[ment2_inx_start:ment2_inx_end]

        # (1, 768)
        span1 = torch.mean(last_hidden1_span, dim=0).reshape(1, -1)
        # (1, 768)
        span2 = torch.mean(last_hidden2_span, dim=0).reshape(1, -1)
        # (1, 1)
        span1_span2 = span1.mm(span2.T)

        # 768 * 2 + 1 = (1, 1537)
        concat_result = torch.cat((span1.reshape(-1), span2.reshape(-1), span1_span2.reshape(-1))).reshape(1, -1)
        gold_label = 1 if mention1.coref_chain == mention2.coref_chain else 0

        batch_result.append(concat_result)
        batch_labels.append(gold_label)

    ret_result = torch.cat(batch_result)
    ret_golds = torch.tensor(batch_labels)
    
    if use_cuda:
        ret_result = ret_result.cuda()
        ret_golds = ret_golds.cuda()

    return ret_result, ret_golds


def mention_feat_to_vec(tokenizer, mention, max_surrounding_contx):
    cntx_before, ment_span, cntx_after = extract_mention_surrounding_context(mention, max_surrounding_contx)

    if len(cntx_before) != 0:
        cntx_before = tokenizer.encode(cntx_before)
    if len(cntx_after) != 0:
        cntx_after = tokenizer.encode(cntx_after)

    ment_span = tokenizer.encode(ment_span)
    sent_tokens = cntx_before + ment_span + cntx_after
    sent_tokens = torch.tensor([sent_tokens])
    return sent_tokens, len(cntx_before), len(cntx_before) + len(ment_span)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Train_Event_gold_mentions.json'
    # _event_validation_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Dev_Event_gold_mentions.json'

    _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Train_Event_gold_mentions.json'
    _event_validation_file = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Dev_Event_gold_mentions.json'

    _model_out = str(LIBRARY_ROOT) + '/saved_models/wiki_trained_model'

    _lr = 2e-5
    _iterations = 4
    _batch_size = 32
    _joint = False
    _type = 'event'
    _alpha = 3
    _use_cuda = True  # args.cuda in ['True', 'true', 'yes', 'Yes']

    _tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    _bert = BertForSequenceClassification.from_pretrained('bert-base-cased',
                                                          output_hidden_states=True,
                                                          output_attentions=True)

    _pairwize_model = PairWiseModel(1537, 250, 2)

    if _use_cuda:
        logger.info(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(1)
        _bert.cuda()
        _pairwize_model.cuda()

    random.seed(1)
    np.random.seed(1)

    _train, _validation = load_datasets(_event_train_file, _event_validation_file, _alpha)
    train_pairwise(_bert, _tokenizer, _pairwize_model, _train, _validation, _batch_size, _iterations, _lr, _use_cuda)
