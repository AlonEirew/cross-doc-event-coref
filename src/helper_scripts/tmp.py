import random

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.dataobjs.topics import Topics, Topic
from src.helper_scripts.extract_wec_tojson import clean_long_mentions
from src.utils import json_utils

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
# _event_sent_unvalid1 = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Dev_Event_gold_mentions_not_validated.json'
# _event_sent_unvalid2 = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Test_Event_gold_mentions_not_validated.json'
# _event_validated_full_clean = str(LIBRARY_ROOT) + '/resources/validated/WEC_Test_Full_Event_gold_mentions_validated.json'
#
# _out_file = str(LIBRARY_ROOT) + '/resources/validated/WEC_Test_Event_gold_mentions_validated2.json'
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

# from src.utils.dataset_utils import convert_to_clusters
#
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
# from src.utils.dataset_utils import convert_to_clusters
#
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
all_mentions = list()
_event1 = str(LIBRARY_ROOT) + '/resources/validated/arie_dev_validated_mentions_v2.json'
_event2 = str(LIBRARY_ROOT) + '/resources/validated/alon_dev_validated_mentions_v2.json'
_event3 = str(LIBRARY_ROOT) + '/resources/validated/oren_test_validated_mentions_v1.json'
all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event1))
all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event2))
all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event3))

good = 0
bad = 0

to_sample_from = list()
for mention in all_mentions:
    if hasattr(mention, "manual_score") and mention.manual_score > -1:
        to_sample_from.append(mention)

sample = random.sample(to_sample_from, 100)
for menton in sample:
    if menton.manual_score >= 4:
        good += 1
    elif 0 < menton.manual_score < 4:
        bad += 1

print("Total=" + str(good + bad))
print("good=" + str(good))
print("bad=" + str(bad))