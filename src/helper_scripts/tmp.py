import random

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.io_utils import write_mention_to_json

# _event_train = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_validated2.json'
# train_mentions = MentionData.read_mentions_json_to_mentions_data_list(_event_train)
# output_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_validated2_10_per.json'
#
# train_clusters = Clusters.from_mentions_to_gold_clusters(train_mentions)
# final_mentions = list()
# for cluster in train_clusters.values():
#     for mention in cluster:
#         if StringUtils.is_verb_phrase(mention.tokens_str):
#             final_mentions.extend(cluster)
#             break
#
# print(str(len(final_mentions)))
# write_mention_to_json(output_file, final_mentions)
# print("Done!")
###################################################################################
# from src.utils.string_utils import StringUtils
#
# event_mentions_file = str(LIBRARY_ROOT) + '/resources/ecb/dev/Event_gold_mentions.json'
# mentions = MentionData.read_mentions_json_to_mentions_data_list(event_mentions_file)
# sample = random.sample(mentions, 100)
# ret_spacy = [str(StringUtils.get_pos_spacy(ment)) for ment in sample]
# ret_nltk = [str(StringUtils.get_pos_nltk(ment)) for ment in sample]
#
# for i, ment in enumerate(sample):
#     print(ment.tokens_str + "=" + ret_spacy[i] + ":" + ret_nltk[i])

# event_mentions_file = str(LIBRARY_ROOT) + '/resources/ecb/dev/Event_gold_mentions.json'
# mentions = MentionData.read_mentions_json_to_mentions_data_list(event_mentions_file)
# fin_mention = list()
# for ment in mentions:
#     ment.mention_head, ment.mention_head_lemma, ment.mention_head_pos, ment.mention_ner = StringUtils.find_head_lemma_pos_ner(ment.tokens_str)
#     fin_mention.append(ment)
#
# output_file = str(LIBRARY_ROOT) + '/resources/ecb/dev/Event_gold_mentions1.json'
# write_mention_to_json(output_file, fin_mention)

###################################################################################
# PERCENT = 80
# _event_train = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_validated2.json'
# train_mentions = MentionData.read_mentions_json_to_mentions_data_list(_event_train)
# output_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_validated2_' + str(PERCENT) + '_per.json'
#
# train_clusters = Clusters.from_mentions_to_gold_clusters(train_mentions)
# cluster_amount = (len(train_clusters) * PERCENT) / 100
# final_mentions = list()
# for cluster in train_clusters.values():
#     if cluster_amount < 0:
#         break
#     cluster_amount -= 1
#     final_mentions.extend(cluster)
#
# print(str(len(final_mentions)))
# write_mention_to_json(output_file, final_mentions)
# print("Done!")

###################################################################################
# PERCENT = 80
# _event_train = str(LIBRARY_ROOT) + '/resources/dataset_full/ecb/train/Event_gold_mentions.json'
# topics_ = Topics()
# topics_.create_from_file(_event_train, keep_order=True)
# output_file = str(LIBRARY_ROOT) + '/resources/dataset_full/ecb/train/Event_gold_mentions_' + str(PERCENT) + '_per.json'
#
# topics_amount = (len(topics_.topics_dict) * PERCENT) / 100
# final_mentions = list()
# for top in topics_.topics_dict.values():
#     if topics_amount < 0:
#         break
#     topics_amount -= 1
#     final_mentions.extend(top.mentions)
#
# print(str(len(final_mentions)))
# write_mention_to_json(output_file, final_mentions)
# print("Done!")

# ecb_pos_pairs = pickle.load(open(str(LIBRARY_ROOT) + "/resources/dataset_full/ecb/train/Event_gold_mentions_PosPairs_Subtopic.pickle", "rb"))
# ecb_neg_pairs = pickle.load(open(str(LIBRARY_ROOT) + "/resources/dataset_full/ecb/train/Event_gold_mentions_NegPairs_Subtopic.pickle", "rb"))
#
# wec_pos_pairs_dev = pickle.load(open(str(LIBRARY_ROOT) + "/resources/dataset_full/wec/dev/Event_gold_mentions_validated2_PosPairs.pickle", "rb"))
# wec_neg_pairs_dev = pickle.load(open(str(LIBRARY_ROOT) + "/resources/dataset_full/wec/dev/Event_gold_mentions_validated2_NegPairs.pickle", "rb"))
#
# wec_pos_pairs_test = pickle.load(open(str(LIBRARY_ROOT) + "/resources/dataset_full/wec/test/Event_gold_mentions_validated2_PosPairs.pickle", "rb"))
# wec_neg_pairs_test = pickle.load(open(str(LIBRARY_ROOT) + "/resources/dataset_full/wec/test/Event_gold_mentions_validated2_NegPairs.pickle", "rb"))
#
#
# all_pos_pairs = ecb_pos_pairs + wec_pos_pairs_dev + wec_pos_pairs_test
# all_neg_pairs = ecb_neg_pairs + random.sample(wec_neg_pairs_dev, int(((len(wec_neg_pairs_dev) * 10) / 100)))
#
# pickle.dump(all_pos_pairs, open(str(LIBRARY_ROOT) + "/resources/dataset_full/ecb/train/Event_gold_mentions_PosPairs_WecDev_ECB.pickle", "w+b"))
# pickle.dump(all_neg_pairs, open(str(LIBRARY_ROOT) + "/resources/dataset_full/ecb/train/Event_gold_mentions_NegPairs_WecDev_ECB.pickle", "w+b"))

###################################################################################

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
# _event_file1 = str(LIBRARY_ROOT) + '/resources/dataset_full/gvc/train/GVC_All_gold_event_mentions.json'
# # output_file = str(LIBRARY_ROOT) + '/resources/wikilinks/Event_gold_mentions_3stem_clean.json'
#
# mentions = MentionData.read_mentions_json_to_mentions_data_list(_event_file1)
#
# final_mentions = dict()
# count = 0
# for mention in mentions:
#     count += 1
#     if mention.mention_head_lemma.lower() not in final_mentions:
#         final_mentions[mention.mention_head_lemma.lower()] = 0
#     final_mentions[mention.mention_head_lemma.lower()] += 1
#
# final_mentions = {k: v for k, v in sorted(final_mentions.items(), key=lambda item: item[1])}
# # print("Before!-" + str(len(mentions)))
# print("Total=" + str(count))
# print("After!-" + str(sum(final_mentions.values())))
# print(str(final_mentions))

# write_mention_to_json(output_file, final_mentions)

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
# # _event1 = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_validated3.json'
# _event2 = str(LIBRARY_ROOT) + '/resources/dataset_full/ecb/dev/Event_gold_mentions.json'
# _event3 = str(LIBRARY_ROOT) + '/resources/dataset_full/ecb/test/Event_gold_mentions.json'
# # all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event1))
# all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event2))
# all_mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_event3))
#
# write_mention_to_json(str(LIBRARY_ROOT) + '/resources/dataset_full/ecb/Event_dev_test_gold_mentions_validated', all_mentions)


# _event1 = str(LIBRARY_ROOT) + '/resources/dataset_full/meantime/train/Event_gold_mentions.json'
# mentions = MentionData.read_mentions_json_to_mentions_data_list(_event1)
# top1 = [item for item in mentions if item.topic_id == 'corpus_airbus']
# print('corpus_airbus=' + str(len(top1)))
#
# top2 = [item for item in mentions if item.topic_id == 'corpus_apple']
# print('corpus_apple=' + str(len(top2)))
#
# top3 = [item for item in mentions if item.topic_id == 'corpus_gm']
# print('corpus_gm=' + str(len(top3)))
#
# top4 = [item for item in mentions if item.topic_id == 'corpus_stock']
# print('corpus_stock=' + str(len(top4)))


# write_mention_to_json(str(LIBRARY_ROOT) + '/resources/dataset_full/meantime/test/Event_gold_mentions.json', flat_list)



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
# train_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#             '/resources/dataset_full/wec/train/Event_gold_mentions_validated2.json')
#
# MAX_CLUSTERS = 150
# MAX_IN_CLUSTER = 60
# to_topics = True
#
# clusters = Clusters.from_mentions_to_gold_clusters(train_mentions)
# type_cluster_count = dict()
# fin_mentions = list()
# for cluster in clusters.values():
#     if len(cluster) <= MAX_IN_CLUSTER:
#         mention_type = cluster[0].mention_type
#         if mention_type not in type_cluster_count:
#             type_cluster_count[mention_type] = 0
#         if type_cluster_count[mention_type] < MAX_CLUSTERS:
#             type_cluster_count[mention_type] += 1
#             if to_topics:
#                 for mention in cluster:
#                     mention.topic_id = mention_type
#
#             fin_mentions.extend(cluster)
#
# write_mention_to_json(
#     str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_limit' + str(MAX_CLUSTERS) + '_topic.json', fin_mentions)
#
# print('Done!')

################# SHAFFEL #####################


# def shuffel_clust(mentions):
#     clusters = Clusters.from_mentions_to_gold_clusters(mentions)
#     clusters = list(clusters.values())
#     random.shuffle(clusters)
#     return clusters
#
#
# def new_index(clust_of_clust):
#     index = 1
#     ret_mentions = list()
#     for clust in clust_of_clust:
#         for ment in clust:
#             ment.mention_index = index
#             ret_mentions.append(ment)
#             index += 1
#
#     return ret_mentions

# train_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#              '/resources/dataset_full/wec/train/Event_gold_mentions_clean11.json')
#
# dev_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#              '/resources/dataset_full/wec/dev/Event_gold_mentions_clean11.json')
#
# test_mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
#              '/resources/dataset_full/wec/test/Event_gold_mentions_clean11.json')
#
# train_clust = shuffel_clust(train_mentions)
# dev_clust = shuffel_clust(dev_mentions)
# test_clust = shuffel_clust(test_mentions)
#
# train_ment_fin = new_index(train_clust)
# dev_ment_fin = new_index(dev_clust)
# test_ment_fin = new_index(test_clust)
#
# write_mention_to_json(str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_clean11.json', train_ment_fin)
# write_mention_to_json(str(LIBRARY_ROOT) + '/resources/dataset_full/wec/dev/Event_gold_mentions_clean11.json', dev_ment_fin)
# write_mention_to_json(str(LIBRARY_ROOT) + '/resources/dataset_full/wec/test/Event_gold_mentions_clean11.json', test_ment_fin)


# calc_singletons(dev_mentions, "dev")
# print("################################")
# calc_singletons(test_mentions, "test")
# print("################################")
# calc_singletons(train_mentions, "train")
# print("################################")
# print("################################")
#
# print("Done!")

#####################################################

# in_all = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/all/Event_gold_mentions_clean8_uncut_span7.json'
# in_splt = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/test/Event_gold_mentions_clean11_validated.json'
#
# out_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/dev/Event_gold_mentions_clean12_validated.json'
#
# mentions_all = MentionData.read_mentions_json_to_mentions_data_list(in_all)
# mentions_splt = MentionData.read_mentions_json_to_mentions_data_list(in_splt)
#
# score = 0
# out_splt = list()
# for ment_splt in mentions_splt:
#     for ment_all in mentions_all:
#         if ment_splt.mention_id == ment_all.mention_id:
#             ment_splt.tokens_str = ment_all.tokens_str
#             ment_splt.tokens_number = ment_all.tokens_number
#             ment_splt.mention_head = ment_all.mention_head
#             ment_splt.mention_head_lemma = ment_all.mention_head_lemma
#             ment_splt.mention_head_pos = ment_all.mention_head_pos
#
#
# write_mention_to_json(out_file, mentions_splt)
# print("Done!")

############################################################

in_all = str(LIBRARY_ROOT) + '/resources/ecb/train/Event_gold_mentions.json'
out_file = str(LIBRARY_ROOT) + '/resources/ecb/train/Event_gold_mentions1.json'
mentions_all = MentionData.read_mentions_json_to_mentions_data_list(in_all)
write_mention_to_json(out_file, mentions_all)
print("Done!")
