
import logging

import numpy as np
import os
import sklearn.cluster
import spacy
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.io_utils import load_json_file

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

nlp = spacy.load('en_core_web_sm')

N_COMPONENTS = 10000


def filter_docs(text):
    parsed = nlp(np.unicode(text))
    filtered_tokens = [tok.text for tok in parsed if tok.ent_iob_ != 'O']
    if len(filtered_tokens) < 3:
        return text
    else:
        return ' '.join(filtered_tokens)


def main(full_context_file, test_mentions_json, predicted_mentions_out_json):
    logger.info('Reading sentences from {}'.format(full_context_file))
    test_topics = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

    true_labels_int = list()
    doc_ids = list()
    sentences = list()
    docs_all = load_json_file(full_context_file)
    for doc_id, sentences_list in docs_all.items():
        topic_id = doc_id.split("_")
        is_plus = True if 'ecbplus' in topic_id[1] else False
        if int(topic_id[0]) in test_topics:
            doc_ids.append(doc_id)
            sentences_as_text = " ".join([" ".join(sentence) for sentence in sentences_list.values()])
            sentences_as_text = filter_docs(sentences_as_text)
            sentences.append(sentences_as_text)
            if is_plus:
                id = int(topic_id[0] + "1")
                true_labels_int.append(id)
            else:
                id = int(topic_id[0] + "0")
                true_labels_int.append(id)

    vectorizer = TfidfVectorizer(max_df=0.5, min_df=3, ngram_range=(1, 3),
                                 stop_words=None)

    vectorized_data = vectorizer.fit_transform(sentences)
    print('Number of documents - {}'.format(len(sentences)))
    logger.info('Clustering to topics...')
    kmeans = sklearn.cluster.KMeans(n_clusters=20, init='k-means++', max_iter=200,
                                    n_init=20, random_state=665,
                                    n_jobs=20, algorithm='auto')
    kmeans.fit(vectorized_data)
    label_mapping = dict()
    for i in range(len(kmeans.labels_)):
        label_mapping[doc_ids[i]] = str(kmeans.labels_[i].item())

    test_mentions = MentionData.read_mentions_json_to_mentions_data_list(test_mentions_json)
    for ment in test_mentions:
        ment.topic_id = label_mapping[ment.doc_id]

    # Evaluation
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels_int, kmeans.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(true_labels_int, kmeans.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(true_labels_int, kmeans.labels_))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(true_labels_int, kmeans.labels_))

    logger.info("writing predicted mentions to file-" + predicted_mentions_out_json)
    # write_mention_to_json(predicted_mentions_out_json, test_mentions)


if __name__ == '__main__':
    main(str(LIBRARY_ROOT) + "/resources/dataset_full/ecb_full_contest_file.json",
         str(LIBRARY_ROOT) + '/resources/dataset_full/ECB_Test_Full_Event_gold_mentions.json',
         str(LIBRARY_ROOT) + '/resources/dataset_full/ECB_Test_Full_Event_gold_predicted_topic_mentions.json')
