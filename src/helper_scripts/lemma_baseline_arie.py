import argparse
from itertools import combinations

import pandas as pd
import random
import spacy
from sklearn.metrics import precision_recall_fscore_support

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='/home/nlp/ariecattan/cross-doc-coref/resources/single_sent_full_context_mean/WEC_Test_Event_gold_mentions.json')
args = parser.parse_args()


def get_positive_negative(mentions):
    positives, negatives = [], []
    grouped_by_topic = mentions.groupby('topic_id')
    print('Number of topics: {}'.format(len(grouped_by_topic)))
    for topic, grouped_mentions_by_topic in grouped_by_topic:
        index_pair = list(combinations(grouped_mentions_by_topic['index'].to_numpy(), 2))
        grouped_by_cluster = grouped_mentions_by_topic.groupby('coref_chain')
        positive_pairs = []
        for cluster, grouped_mentions_by_cluster in grouped_by_cluster:
            positive_pairs.extend(list(combinations(grouped_mentions_by_cluster['index'].to_numpy(), 2)))
        negative_pairs = set(index_pair) - set(positive_pairs)


        negatives.extend(list(negative_pairs))
        positives.extend(positive_pairs)

    return positives, negatives





if __name__ == '__main__':
    print('File: {}'.format(args.input_file))
    event_mentions = pd.read_json(args.input_file).reset_index()
    # event_mentions['topic'] = [re.search(r"\b(\d+)\D+", str(x)).group(1) for x in event_mentions['topic_id']]
    event_mentions['mention_lemma'] = [[token.lemma_ for token in nlp(mention['tokens_str'])] for i, mention in event_mentions.iterrows()]

    positives, negatives = get_positive_negative(event_mentions)
    negatives = random.sample(negatives, len(positives) * 25)
    labels = [True] * len(positives) + [False] * len(negatives)
    print('Positive/Negative: {}/{}'.format(len(positives), len(negatives)))

    exact_string = [mention['tokens_str'] for i, mention in  event_mentions.iterrows()]
    lemmas = [mention['mention_lemma'] for i, mention in  event_mentions.iterrows()]

    preds = [True if lemmas[id1] == lemmas[id2] else False for id1, id2 in positives + negatives]
    exact_preds = [True if exact_string[id1] == exact_string[id2] else False for id1, id2 in positives + negatives]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, pos_label=True, average='binary')
    print('Same lemma - Precision/Recall/F1 : {}/{}/{}'.format(precision, recall, f1))
    precision, recall, f1, _ = precision_recall_fscore_support(labels, exact_preds, pos_label=True, average='binary')
    print('Exact string - Precision/Recall/F1 : {}/{}/{}'.format(precision, recall, f1))
