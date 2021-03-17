"""
Usage:
    stats_calculation.py <MentionsFile>

Options:
    -h --help     Show this screen.

"""

from collections import namedtuple, Counter

from docopt import docopt
from transformers import RobertaTokenizer

from src.dataobjs.cluster import Clusters
from src.dataobjs.mention_data import MentionData
from src.utils.embed_utils import EmbedTransformersGenerics
from src.utils.string_utils import StringUtils


def count_verb_mentions(split_list):
    verb_phrases = 0
    for mention in split_list:
        if StringUtils.is_verb_phrase(mention.tokens_str):
            verb_phrases += 1

    print("Total verb phrases=" + str(verb_phrases))


def calc_longest_mention_and_context(split_list):
    longest_mention = 0
    longest_context = 0
    for mention in split_list:
        mention_encode = _tokenizer.encode(mention.mention_context[mention.tokens_number[0]:mention.tokens_number[-1] + 1])
        context_encode = EmbedTransformersGenerics.extract_mention_surrounding_context(mention, 205)
        if len(mention_encode) > longest_mention:
            longest_mention = len(mention_encode)
        if len(mention.mention_context) > longest_context:
            longest_context = len(context_encode)

    print('Longest mention span=' + str(longest_mention))
    print('Longest_context=' + str(longest_context))


def produce_cluster_stats(clusters):
    singletons_count = 0
    sum_mentions = 0
    sum_mentions_no_single = 0
    all_lemmas = list()
    all_lemmas_no_single = list()
    same_string_in_cluster = dict()
    print('Clusters=' + str(len(clusters)))
    biggest_cluster = 0
    for index, clust in enumerate(clusters.values()):
        clust_len = len(clust)
        clust_lemmas = set([ment.mention_head_lemma for ment in clust])
        cluster_uniqe_str = Counter([ment.tokens_str for ment in clust])
        same_string_in_cluster[index] = sum(cluster_uniqe_str.values()) / len(cluster_uniqe_str)

        if clust_len == 1:
            singletons_count += 1
        else:
            sum_mentions_no_single += clust_len
            all_lemmas_no_single.extend(clust_lemmas)

        if len(clust) > biggest_cluster:
            biggest_cluster = clust_len

        sum_mentions += len(clust)
        all_lemmas.extend(clust_lemmas)

    print('Singletons=' + str(singletons_count))
    print('Non_singleton_Clusters=' + str(len(clusters) - singletons_count))
    print('Biggest cluster=' + str(biggest_cluster))
    print('Average Ment in Clust (include singletons)=' + str(sum_mentions / len(clusters)))
    print('Average Ment in Clust (exclude singletons)=' + str(sum_mentions_no_single / (len(clusters) - singletons_count)))
    print('Average Lemmas in Clust (Diversity-include singletons)=' + str(len(all_lemmas) / len(clusters)))
    print('Average Lemmas in Clust (Diversity-exclude singletons)=' + str(len(all_lemmas_no_single) / (len(clusters) - singletons_count)))
    print('Average Mentions with Same String in Clust=' + str(sum(same_string_in_cluster.values()) / (len(same_string_in_cluster))))


def calc_single_head_lemma_cluster(ment_list, clus_size_thresh):
    clusters = Clusters.from_mentions_to_gold_clusters(ment_list)
    produce_cluster_stats(clusters)
    lemma_clust = dict()
    for clust_id, cluster in clusters.items():
        if len(cluster) > clus_size_thresh:
            if clust_id not in lemma_clust:
                lemma_clust[clust_id] = dict()
            for ment in cluster:
                ment_key = ment.mention_head_lemma.lower()
                if not ment_key in lemma_clust[clust_id]:
                    lemma_clust[clust_id][ment_key] = 0
                lemma_clust[clust_id][ment_key] += 1

    diverse_clusts = list()
    single_head_lemma_clust = 0
    for key, head_set in lemma_clust.items():
        if len(head_set) == 1:
            single_head_lemma_clust += 1
        else:
            diverse_clusts.append(head_set)

    print("Single head lemma clusters=" + str(single_head_lemma_clust))


def calc_dist_lemmas_cross(split_list):
    print('Mentions=' + str(len(split_list)))
    mention_length_sum = sum([len(ment.tokens_number) for ment in split_list])
    average_length = mention_length_sum / len(split_list)
    print('Average Ment Length (tokens)=' + str(average_length))

    distinct_lemmas = dict()
    distinct_lemmas_cross = dict()
    for mention in split_list:
        if mention.mention_head_lemma.lower() not in distinct_lemmas:
            distinct_lemmas[mention.mention_head_lemma.lower()] = mention

        lem_id = mention.mention_head_lemma.lower() + '_' + ''.join(filter(lambda i: i.isdigit(), str(mention.topic_id)))
        if lem_id not in distinct_lemmas_cross:
            distinct_lemmas_cross[lem_id] = set()
        distinct_lemmas_cross[lem_id].add(mention.coref_chain)

    sum_cross_clust_lem = sum([1 for clust_set in distinct_lemmas_cross.values() if len(clust_set) > 1])
    avg_cross_clust_lem = sum([len(clust_set) for clust_set in distinct_lemmas_cross.values()]) / len(distinct_lemmas_cross)

    print('Distinct Lemmas in corpus=' + str(len(distinct_lemmas)))
    print('Distinct Lemmas across clusters=' + str(sum_cross_clust_lem))
    print('Avg num of clusters with same Lemma(Ambiguity)=' + str(avg_cross_clust_lem))
    print()

    count_verb_mentions(split_list)
    # json.dump({k: v for k, v in sorted(distinct_lemmas.items(), key=lambda item: item[1])}, sys.stdout)
    print()


def cross_doc_clusters(mentions):
    MentionKey = namedtuple("MentKey", ["coref", "doc_id"])
    cross_doc_clust = dict()
    documents = set()
    for ment in mentions:
        if ment.doc_id not in documents:
            documents.add(ment.doc_id)

        ment_key = MentionKey(ment.coref_chain, ment.doc_id)
        if ment_key not in cross_doc_clust:
            cross_doc_clust[ment_key] = list()
        cross_doc_clust[ment_key].append(ment)

    cross_clusters = dict()
    for coref1, doc_id1 in cross_doc_clust.keys():
        if coref1 not in cross_clusters:
            cross_clusters[coref1] = 0
        cross_clusters[coref1] += 1

    print("Documents = " + str(len(documents)))
    print("Cross Doc Clusters = " + str(sum(1 for value in cross_clusters.values() if value > 1)))


def generate_pair_score(f):
    same_string = 0
    diff_string = 0
    total = 0
    for line in f.readlines():
        split = line.split("=")
        if len(split) == 2:
            total += 1
            if split[0].strip().lower() == split[1].strip().lower():
                same_string += 1
            else:
                diff_string += 1
    f.close()
    return diff_string, same_string, total


def create_split_stats():
    mentions_list = MentionData.read_mentions_json_to_mentions_data_list(_mention_file)
    if mentions_list:
        print('############# ' + _mention_file + ' ###################')
        calc_dist_lemmas_cross(mentions_list)
        calc_longest_mention_and_context(mentions_list)
        cross_doc_clusters(mentions_list)
        calc_single_head_lemma_cluster(mentions_list, 1)


if __name__ == '__main__':
    _arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    print(_arguments)
    _mention_file = _arguments.get("<MentionsFile>")
    _tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    create_split_stats()
