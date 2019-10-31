import logging
import random

import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from torch import nn
from transformers import BertForSequenceClassification, AdamW, BertTokenizer
from tqdm import trange

from src import LIBRARY_ROOT
from src.utils.dl_utils import get_feat

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def finetune_bert(bert, tokenizer, train, validation, batch_size, use_cuda):
    linear = nn.Linear(100, 2)

    loss_fun = nn.CrossEntropyLoss()
    param_optimizer = list(bert.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 4

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        bert.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        dataset_size = len(train)
        end_index = batch_size
        # Train the data for one epoch
        for start_index in range(0, dataset_size, batch_size):
            if end_index > dataset_size:
                end_index = dataset_size

            batch_features = train[start_index:end_index].copy()

            batch, att_mask, true_label = feat_to_vec(tokenizer, batch_features, use_cuda)
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss = bert(batch, attention_mask=att_mask, labels=true_label)[0]
            # final = linear(output)
            # loss = loss_fun(output, true_label)

            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            # nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            end_index += batch_size

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        bert.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        # for batch in validation_dataloader:
        #     # Add batch to GPU
        #     batch = tuple(t.to(device) for t in batch)
        #     # Unpack the inputs from our dataloader
        #     b_input_ids, b_input_mask, b_labels = batch
        #     # Telling the model not to compute or store gradients, saving memory and speeding up validation
        #     with torch.no_grad():
        #         # Forward pass, calculate logit predictions
        #         logits = bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        #
        #     # Move logits and labels to CPU
        #     logits = logits.detach().cpu().numpy()
        #     label_ids = b_labels.to('cpu').numpy()
        #
        #     tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        #
        #     eval_accuracy += tmp_eval_accuracy
        #     nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))


def feat_to_vec(tokenizer, batch_features, use_cuda):
    MAX_LEN = 256
    ret_golds = list()
    input_ids_list = list()
    att_mask_list = list()
    for mention1, mention2 in batch_features:
        sentence1_words = '[CLS] ' + ' '.join(mention1.mention_context)
        sent1_tokens = tokenizer.tokenize(sentence1_words)
        att_mask = [1] * len(sent1_tokens)

        sentence2_words = ' [SEP]' + ' '.join(mention2.mention_context) + ' [SEP]'
        sent2_tokens = tokenizer.tokenize(sentence2_words)
        att_mask.extend([1] * len(sent2_tokens))

        pair_tokens = sent1_tokens + sent2_tokens
        att_mask.extend([0] * (MAX_LEN - len(pair_tokens)))
        input_ids = tokenizer.convert_tokens_to_ids(pair_tokens)

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


def create_dataloader(train_file, validation_file):
    logger.info('Create Features:')
    train_feat = get_feat(train_file)
    validation_feat = get_feat(validation_file)
    return train_feat, validation_feat


if __name__ == '__main__':
    _event_train_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Small_Event_gold_mentions.json'
    _event_validation_file = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Small_Event_gold_mentions.json'

    _model_out = str(LIBRARY_ROOT) + '/saved_models/wiki_trained_model'

    _learning_rate = 0.01
    _iterations = 1
    _batch_size = 1
    _joint = False
    _type = 'event'
    _use_cuda = False  # args.cuda in ['True', 'true', 'yes', 'Yes']

    _bert = BertForSequenceClassification.from_pretrained('bert-large-cased')
    _tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

    if _use_cuda:
        logger.info(torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(1)
        _bert.cuda()

    random.seed(1)
    np.random.seed(1)

    _train, _validation = create_dataloader(_event_train_file, _event_validation_file)
    finetune_bert(_bert, _tokenizer, _train, _validation, _batch_size, _use_cuda)
