import pickle

import ntpath
import re
from os import listdir

import random
from sklearn import metrics

from src import LIBRARY_ROOT
from src.dataobjs.cluster import Clusters
from src.dataobjs.dataset import Split
from src.dataobjs.mention_data import MentionData
from src.utils.embed_utils import EmbedFromFile
from src.utils.string_utils import StringUtils
from src.dataobjs.topics import Topics, Topic
from src.helper_scripts.extract_wec_tojson import clean_long_mentions, extract_from_sql
from src.utils.io_utils import load_pickle, write_mention_to_json
from src.utils.sqlite_utils import create_connection, select_from_validation, select_all_from_mentions

# all_mentions = list()
# _event_file1 = str(LIBRARY_ROOT) + '/resources/validated/WEC_Test_Event_gold_mentions_validated.json'
# _event_file2 = str(LIBRARY_ROOT) + '/resources/validated/oren_test_validated_mentions_v1.json'
# _event_file3 = str(LIBRARY_ROOT) + '/resources/validated/validation_for_test.json'
# _event_file4 = str(LIBRARY_ROOT) + '/resources/validated/alon_test_validated_mentions_v1.json'
#
# _out_file = str(LIBRARY_ROOT) + '/resources/validated/WEC_Test_Full_Event_gold_mentions_validated.json'
#
# all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event_file1))
# all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event_file2))
# all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event_file3))
# all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event_file4))
#
# dups = set()
# final_mentions = list()
# for mention in all_mentions:
#     # if hasattr(mention, "manual_score") and mention.manual_score >= 4:
#     if mention.mention_id in dups:
#         print("DUP!!!!")
#     else:
#         final_mentions.append(mention)
#         dups.add(mention.mention_id)
#
# json_utils.write_mention_to_json(_out_file, final_mentions)
# print("DONE!!! " + str(len(final_mentions)))
###################################################################################
# all_mentions = list()
# _event_sent_unvalid1 = str(LIBRARY_ROOT) + '/resources/tmp/WEC_Dev_Full_Event_gold_mentions_not_validated.json'
# _event_sent_unvalid2 = str(LIBRARY_ROOT) + '/resources/tmp/WEC_Test_Full_Event_gold_mentions_not_validated.json'
# _event_validated_full_clean = str(LIBRARY_ROOT) + '/resources/tmp/WEC_Dev_Event_gold_mentions_validated.json'
#
# _out_file = str(LIBRARY_ROOT) + '/resources/tmp/WEC_Dev_Full_Event_gold_mentions_validated.json'
#
# un_validated = MentionData.read_mentions_json_to_mentions_data_list(_event_sent_unvalid1)
# un_validated.extend(MentionData.read_mentions_json_to_mentions_data_list(_event_sent_unvalid2))
# validated = MentionData.read_mentions_json_to_mentions_data_list(_event_validated_full_clean)
#
#
# final_mentions = list()
# for mention_val in validated:
#     for mention_unval in un_validated:
#         if str(mention_unval.mention_id) == str(mention_val.mention_id):
#             final_mentions.append(mention_unval)
#             break
#
# json_utils.write_mention_to_json(_out_file, final_mentions)
#
# print("DONE!!! " + str(len(final_mentions)))
####################################################################################
# all_mentions = list()
# _event_file1 = str(LIBRARY_ROOT) + '/resources/validated/WEC_CLEAN_JOIN_ALL.json'
# _event_file2 = str(LIBRARY_ROOT) + '/resources/validated/WEC_Dev_Event_gold_mentions.json'
#
# group1 = MentionData.read_mentions_json_to_mentions_data_list(_event_file1)
# group2 = MentionData.read_mentions_json_to_mentions_data_list(_event_file2)
#
# keys1 = dict()
# for mention in group1:
#     if not mention.mention_id in keys1:
#         keys1[mention.mention_id] = 1
#     else:
#         keys1[mention.mention_id] += 1
#
# keys2 = dict()
# for mention in group2:
#     if not mention.mention_id in keys2:
#         keys2[mention.mention_id] = 1
#     else:
#         keys2[mention.mention_id] += 1
#
# for key, value in keys1.items():
#     if value > 1:
#         print("keys1" + key)
#
# for key, value in keys2.items():
#     if value > 1:
#         print("keys2" + key)
#
# print(set(keys1.keys()).difference(keys2.keys()))
####################################################################################
# _event_file1 = str(LIBRARY_ROOT) + '/resources/validated/WEC_Test_Event_gold_mentions_validated.json'
# # output_file = str(LIBRARY_ROOT) + '/resources/validated/WEC_Test_Event_gold_mentions_validated.json'
#
# mentions = MentionData.read_mentions_json_to_mentions_data_list(_event_file1)
# final_mentions = clean_long_mentions(mentions)
# print("DONE!-" + str(len(final_mentions)))

# json_utils.write_mention_to_json(output_file, final_mentions)
####################################################################################
# all_mentions = list()
# _event_file1 = str(LIBRARY_ROOT) + '/resources/validated/validation_for_test.json'
# _event_file2 = str(LIBRARY_ROOT) + '/resources/validated/oren_test_validated_mentions_v1.json'
# mentions = MentionData.read_mentions_json_to_mentions_data_list(_event_file2)
#
# out_file = str(LIBRARY_ROOT) + '/resources/validated/alon_test_validated_mentions_v1.json'
# new_validation = str(LIBRARY_ROOT) + '/resources/validated/validation_for_test2.json'
#
# topics_ = Topics()
# topics_.create_from_file(_event_file1, keep_order=True)
# clusters = convert_to_clusters(topics_)
#
# for mention in mentions:
#     if hasattr(mention, "manual_score"):
#         if mention.coref_chain in clusters:
#             del clusters[mention.coref_chain]
#
# final_mentions_val = list()
# final_mentions_unval = list()
# for clut in clusters.values():
#     for ment in clut:
#         if hasattr(ment, "manual_score"):
#             final_mentions_val.append(ment)
#         else:
#             final_mentions_unval.append(ment)
#
# json_utils.write_mention_to_json(out_file, final_mentions_val)
# json_utils.write_mention_to_json(new_validation, final_mentions_unval)

################### SPLIT ########################
# start = 0
# end = 100
# i = 1
# while end < len(final_clust) and end <= 500:
#     set1 = list()
#     for clust in final_clust[start:end]:
#         for ment in clust:
#             set1.append(ment)
#
#     start = end
#     end += 100
#     print("Done !")
#     json_utils.write_mention_to_json(str(LIBRARY_ROOT) +"/resources/validated/for_validation_mentions_set" + str(i) + ".json", set1)
#     i += 1
################### SPLIT ########################
################### Create CoNLL file ########################
# _event_file1 = str(LIBRARY_ROOT) + '/resources/validated/WEC_Test_Full_Event_gold_mentions_reduced.json'
# output_file = str(LIBRARY_ROOT) + "/resources/gold_scorer/wec/CD_test_event_reduced_mention_based.txt"
# mentions = MentionData.read_mentions_json_to_mentions_data_list(_event_file1)
#
# """
# :param mentions: List[MentionData]
# :param output_file: str
# :return:
# """
# output = open(output_file, 'w')
# output.write('#begin document (ECB+/ecbplus_all); part 000\n')
# for mention in mentions:
#     output.write('ECB+/ecbplus_all\t' + '(' + str(mention.coref_chain) + ')\n')
# output.write('#end document')
# output.close()
# print(str(len(mentions)))
##################################################
####def remove_30_cluster_single_head_lemma(ment_file, message, clus_size_thresh)#####
# ment_file = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Test_Event_gold_mentions.json'
# topics = Topics()
# topics.create_from_file(ment_file)
# clusters = convert_to_clusters(topics)
# lemma_clust = dict()
# singletons_mentions = list()
# for clust in clusters.items():
#     if len(clust[1]) > 1:
#         if clust[0] not in lemma_clust:
#             lemma_clust[clust[0]] = dict()
#         for ment in clust[1]:
#             ment_key = ment.mention_head_lemma.lower()
#             if not ment_key in lemma_clust[clust[0]]:
#                 lemma_clust[clust[0]][ment_key] = 0
#             lemma_clust[clust[0]][ment_key] += 1
#     else:
#         singletons_mentions.extend(clust[1])
#
# final_results = list()
# single_head_lemma_clust = 0
# for key, head_set in lemma_clust.items():
#     if len(head_set) == 1:# and single_head_lemma_clust < 100 and bool(random.getrandbits(1)):
#         single_head_lemma_clust += 1
#     else:
#         final_results.extend(clusters[key])
#
# final_results.extend(singletons_mentions)
# print(str(single_head_lemma_clust))
# output_file = str(LIBRARY_ROOT) + '/resources/validated/WEC_Test_Full_Event_gold_mentions_reduced.json'
# json_utils.write_mention_to_json(output_file, final_results)
# print(str(len(final_results)))
# ###############################################
# all_mentions = list()
# _event1 = str(LIBRARY_ROOT) + '/resources/validated/arie_dev_validated_mentions_v2.json'
# _event2 = str(LIBRARY_ROOT) + '/resources/validated/alon_dev_validated_mentions_v2.json'
# _event3 = str(LIBRARY_ROOT) + '/resources/validated/oren_test_validated_mentions_v1.json'
# all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event1))
# all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event2))
# all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event3))
#
# good = 0
# bad = 0
#
# to_sample_from = list()
# for mention in all_mentions:
#     if hasattr(mention, "manual_score") and mention.manual_score > -1:
#         to_sample_from.append(mention)
#
# sample = random.sample(to_sample_from, 100)
# for menton in sample:
#     if menton.manual_score >= 4:
#         good += 1
#     elif 0 < menton.manual_score < 4:
#         bad += 1
#
# print("Total=" + str(good + bad))
# print("good=" + str(good))
# print("bad=" + str(bad))
################################### Load Predicted From File #########################################
# test_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#             '/resources/dataset_full/ecb/test/Event_gold_mentions.json')
#
# label_mapping = dict()
# pred_topics = load_pickle(str(LIBRARY_ROOT) + '/resources/ecb_pred/shany_predicted_topics')
#
# for i in range(len(pred_topics)):
#     for doc_name in pred_topics[i]:
#         label_mapping[doc_name] = i
#
# docs_ids = dict()
# for ment in test_mentions:
#     doc_id_split = ment.doc_id.split('.')[0]
#     if doc_id_split not in docs_ids:
#         docs_ids[doc_id_split] = ment.topic_id
#
#     topic_id = ment.doc_id.split("_")
#     ment.topic_id = label_mapping[doc_id_split]
#
# true_labels_int = list()
# pred_labels_int = list()
# for doc_id in docs_ids.keys():
#     is_plus = True if 'ecbplus' in doc_id else False
#     if is_plus:
#         id = int(doc_id.split('_')[0] + "1")
#         true_labels_int.append(id)
#     else:
#         id = int(doc_id.split('_')[0] + "0")
#         true_labels_int.append(id)
#
#     pred_labels_int.append(label_mapping[doc_id])
#
# for key in label_mapping.keys():
#     if key not in docs_ids:
#         print(key + " not found")
#
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels_int, pred_labels_int))
# print("Completeness: %0.3f" % metrics.completeness_score(true_labels_int, pred_labels_int))
# print("V-measure: %0.3f" % metrics.v_measure_score(true_labels_int, pred_labels_int))
# print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(true_labels_int, pred_labels_int))
# write_mention_to_json(str(LIBRARY_ROOT) + '/resources/dataset_full/ecb/test/Event_shany_pred_mentions.json', test_mentions)

######################## CLEAN NER #############################
# embed_utils = EmbedFromFile([str(LIBRARY_ROOT) + "/resources/dataset_full/ecb/train/Event_gold_mentions_roberta-large.pickle"], 1024)
#
# split = Split.Dev.name.lower()
# origin_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#                        '/resources/dataset_full/wec/' + split + '/Event_gold_mentions_validated.json')
# new_mention = list()
# removed = 0
# for mention in origin_mentions:
#     ners = StringUtils.find_all_ners(mention.mention_context)
#
#     head_token_id = -1
#     for i in mention.tokens_number:
#         if mention.mention_context[i] == mention.mention_head:
#             head_token_id = i
#
#     if head_token_id == -1:
#         removed += 1
#         print("Bad mention in split-" + str(mention.mention_id))
#         # raise Exception("Couldnt find mention head in context")
#     # else:
#     #     ment_ners = ners[head_token_id]
#     #     if ment_ners in ["GPE", "LOC"]:
#     #         removed += 1
#     #     else:
#     #         new_mention.append(mention)
#
# print("total mentions remove=" + str(removed))
# print("total mentions in split-" + str(len(new_mention)))
# write_mention_to_json(str(LIBRARY_ROOT) + '/resources/dataset_full_clean/wec/' + split + '/Event_gold_mentions_validated.json', new_mention)

###################### Add Context and Topic Id to Shany file ##################
# shany_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#             '/gold_scorer/shany_ecb/ECB_Test_Event_gold_mentions.json')
#
# test_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#             '/resources/dataset_full/ecb/test/Event_gold_mentions.json')
#
# for mention_s in shany_mentions:
#     mention_s.doc_id = mention_s.doc_id + ".xml"
#     mention_s.mention_id = None
#     mention_s.get_mention_id()
#     for mention_t in test_mentions:
#         if mention_s.mention_id == mention_t.mention_id:
#             # mention_s.mention_context = mention_t.mention_context
#             mention_s.topic_id = mention_t.topic_id
#             # mention_s.tokens_number = mention_t.tokens_number
#             break
#
# for mention_s in shany_mentions:
#     if not mention_s.topic_id or len(mention_s.topic_id) == 0:
#         raise Exception("Missing context in mention-" + mention_s.topic_id)
#
# write_mention_to_json(str(LIBRARY_ROOT) + '/gold_scorer/shany_ecb/ECB_Test_Event_gold_mentions_context.json', shany_mentions)
# print("Done!")
##########################################################################################

# dev_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#             '/resources/dataset/wec/dev/WEC_Dev_Event_gold_mentions_validated.json')
#
# test_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#             '/resources/dataset/wec/test/WEC_Test_Event_gold_mentions_validated.json')
#
# train_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#             '/resources/dataset/wec/train/WEC_Train_Event_gold_mentions_validated.json')
#
# connection = create_connection(str(LIBRARY_ROOT) + "/resources/EnWikiLinks_v9.db")
# # clusters = select_all_from_mentions(connection, 'Validation3', limit=-1)
# mentions_full_context, _ = extract_from_sql(connection, "Validation3", "split")
# all_mentions = dev_mentions + test_mentions + train_mentions
#
# print("Total mentions from files=" + str(len(all_mentions)))
# print("Total mentions from sql=" + str(len(mentions_full_context)))
#
# mentions_by_ids = dict()
# for mention_from_file in all_mentions:
#     if mention_from_file.mention_id in mentions_by_ids:
#         raise Exception("Mention - " + mention_from_file.mention_id + " exist twice!!")
#     mentions_by_ids[mention_from_file.mention_id] = mention_from_file
#
# total_changed = 0
# for mention_from_sql in mentions_full_context:
#     if mention_from_sql.mention_id in mentions_by_ids:
#         total_changed += 1
#         mentions_by_ids[mention_from_sql.mention_id].mention_context = mention_from_sql.mention_context
#         mentions_by_ids[mention_from_sql.mention_id].tokens_number = mention_from_sql.tokens_number
#
# if total_changed < len(all_mentions):
#     print("Total changed is smaller then the sum of all mentions: " + str(total_changed) + "<" + str(len(all_mentions)))
# else:
#     print('Writing files...')
#     write_mention_to_json(str(LIBRARY_ROOT) + '/resources/dataset_full/wec/dev/WEC_Full_Dev_Event_gold_mentions_validated.json',
#                           dev_mentions)
#
#     write_mention_to_json(str(LIBRARY_ROOT) + '/resources/dataset_full/wec/test/WEC_Full_Test_Event_gold_mentions_validated.json',
#                           test_mentions)
#
#     write_mention_to_json(str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/WEC_Full_Train_Event_gold_mentions_validated.json',
#                           train_mentions)
#
# print('Done!!')
#######################################################################
############################### REMOVE SINGLETONS ###############################
# topics_ = Topics()
# topics_.create_from_file(
#     str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_limit500_validated2.json', keep_order=True)
# clusters = topics_.convert_to_clusters()
#
# fin_mentions = list()
# for ments_list in clusters.values():
#     if len(ments_list) > 1:
#         fin_mentions.extend(ments_list)
#
# write_mention_to_json(
#     str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_limit500.json', fin_mentions)
#############################################################################################

train_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
            '/resources/dataset_full/wec/train/Event_gold_mentions_validated2.json')

MAX_CLUSTERS = 150
MAX_IN_CLUSTER = 60
to_topics = True

clusters = Clusters.from_mentions_to_gold_clusters(train_mentions)
type_cluster_count = dict()
fin_mentions = list()
for cluster in clusters.values():
    if len(cluster) <= MAX_IN_CLUSTER:
        mention_type = cluster[0].mention_type
        if mention_type not in type_cluster_count:
            type_cluster_count[mention_type] = 0
        if type_cluster_count[mention_type] < MAX_CLUSTERS:
            type_cluster_count[mention_type] += 1
            if to_topics:
                for mention in cluster:
                    mention.topic_id = mention_type

            fin_mentions.extend(cluster)

write_mention_to_json(
    str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_limit' + str(MAX_CLUSTERS) + '_topic.json', fin_mentions)

print('Done!')
