import logging

from src import LIBRARY_ROOT
from src.obj.topics import Topics

logger = logging.getLogger(__name__)


def main():
    corpus_id = 'wiki'

    cluster_int = 1
    coref_map = dict()
    in_file_test = str(LIBRARY_ROOT) + '/resources/corpora/' + corpus_id + '/gold_json/WIKI_Dev_Event_gold_mentions.json'
    out_file_test = str(LIBRARY_ROOT) + '/resources/corpora/' + corpus_id + '/gold_scorer/WIKI_CD_dev_entity_based.txt'

    topics = Topics()
    topics.create_from_file(in_file_test)

    output = open(out_file_test, 'w')
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
            output.write('ECB+/ecbplus_all\t' + '(' + str(coref_t) + ')\n' )
    output.write('#end document')


if __name__ == '__main__':
    main()
