import logging

from src import LIBRARY_ROOT
from src.obj.topics import Topics

logger = logging.getLogger(__name__)


def main():
    corpus_id = 'wiki'
    dev_in_file = str(LIBRARY_ROOT) + '/resources/corpora/' + corpus_id + '/gold_json/WIKI_Dev_Event_gold_mentions.json'
    dev_out_file = str(LIBRARY_ROOT) + '/resources/corpora/' + corpus_id + '/gold_scorer/WIKI_CD_dev_event_mention_based.txt'

    test_in_file = str(LIBRARY_ROOT) + '/resources/corpora/' + corpus_id + '/gold_json/WIKI_Test_Event_gold_mentions.json'
    test_out_file = str(LIBRARY_ROOT) + '/resources/corpora/' + corpus_id + '/gold_scorer/WIKI_CD_test_event_mention_based.txt'

    train_in_file = str(LIBRARY_ROOT) + '/resources/corpora/' + corpus_id + '/gold_json/WIKI_Train_Event_gold_mentions.json'
    train_out_file = str(LIBRARY_ROOT) + '/resources/corpora/' + corpus_id + '/gold_scorer/WIKI_CD_train_event_mention_based.txt'

    make_kian_scorer_file(dev_in_file, dev_out_file)
    make_kian_scorer_file(test_in_file, test_out_file)
    make_kian_scorer_file(train_in_file, train_out_file)


def make_kian_scorer_file(in_file, out_file):
    cluster_int = 1
    coref_map = dict()
    topics = Topics()
    topics.create_from_file(in_file)
    output = open(out_file, 'w')
    output.write('#begin document (ECB+/ecbplus_all); part 000\n')
    for topic in topics.topics_list:
        for mention in topic.mentions:
            coref_t = None
            if mention.coref_chain in coref_map:
                coref_t = coref_map[mention.coref_chain]
            else:
                coref_t = cluster_int
                coref_map[mention.coref_chain] = coref_t
                cluster_int += 1
            output.write('ECB+/ecbplus_all\t' + '(' + str(coref_t) + ')\n')
    output.write('#end document')


if __name__ == '__main__':
    main()
