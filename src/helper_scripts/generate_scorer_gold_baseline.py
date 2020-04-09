import logging

from src import LIBRARY_ROOT
from src.dataobjs.dataset import WecDataSet, Split
from src.dataobjs.topics import Topics

logger = logging.getLogger(__name__)


def main():
    dataset = "dataset_full"
    corpus_id = WecDataSet()

    dev_in_file = str(LIBRARY_ROOT) + '/resources/' + dataset + '/' + corpus_id.name.lower() + \
                  '/dev/Event_gold_mentions_validated2.json'
    dev_out_file = str(LIBRARY_ROOT) + '/gold_scorer/' + corpus_id.name.lower() + '/CD_dev_event_mention_based_dataset.txt'

    test_in_file = str(LIBRARY_ROOT) + '/resources/' + dataset + '/' + corpus_id.name.lower() + \
                   '/test/Event_gold_mentions_validated2.json'
    test_out_file = str(LIBRARY_ROOT) + '/gold_scorer/' + corpus_id.name.lower() + '/CD_test_event_mention_based_dataset.txt'

    print("preparing dev file-" + dev_in_file)
    make_kian_scorer_file(dev_in_file, dev_out_file)
    print("preparing test file-" + test_in_file)
    make_kian_scorer_file(test_in_file, test_out_file)


def make_kian_scorer_file(in_file, out_file):
    cluster_int = 1
    coref_map = dict()
    topics = Topics()
    topics.create_from_file(in_file, False)
    output = open(out_file, 'w')
    output.write('#begin document (ECB+/ecbplus_all); part 000\n')
    for topic in topics.topics_dict.values():
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
