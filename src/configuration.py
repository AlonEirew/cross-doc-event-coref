from src import LIBRARY_ROOT
from src.coref_system.relation_extraction import RelationTypeEnum
from src.dataobjs.dataset import DATASET_NAME, SPLIT
from src.utils.clustering_utils import ClusteringType
from src.utils.embed_utils import EmbeddingConfig, EmbeddingEnum

########################## Train Model Params ################################
train_learning_rate = 1e-4
train_batch_size = 32
train_ratio = -1
train_iterations = 1
train_use_cuda = True
train_save_model = True
train_save_model_threshold = 0.1
train_fine_tune = False
train_weight_decay = 0.01
train_hidden_n = 150

train_context_set = "dataset_full"
train_dataset = DATASET_NAME.ECB
dev_dataset = DATASET_NAME.ECB
train_embed_config = EmbeddingConfig(EmbeddingEnum.ROBERTA_LARGE)

train_save_model_file = str(LIBRARY_ROOT) + "/saved_models/" + train_dataset.name + "_" + dev_dataset.name + \
                        "_Test_" + train_embed_config.model_name + "_" + str(train_ratio)

train_load_model_file = str(LIBRARY_ROOT) + "/saved_models/WEC_WEC_200320_bert_large_35iter_18"

train_event_train_file_pos = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                             train_dataset.name.lower() + "/train/Event_gold_mentions_PosPairs_Subtopic.pickle"
train_event_train_file_neg = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                             train_dataset.name.lower() + "/train/Event_gold_mentions_NegPairs_Subtopic.pickle"
train_event_validation_file_pos = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                                  dev_dataset.name.lower() + "/dev/Event_gold_mentions_PosPairs_Subtopic.pickle"
train_event_validation_file_neg = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                                  dev_dataset.name.lower() + "/dev/Event_gold_mentions_NegPairs_Subtopic.pickle"

train_embed_files = [str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + train_dataset.name.lower() +
                     "/train/Event_gold_mentions_" + train_embed_config.model_name + ".pickle",
                     str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + dev_dataset.name.lower() +
                     "/dev/Event_gold_mentions_" + train_embed_config.model_name + ".pickle"]

########################## Inference Model Params ################################
inference_context_set = "dataset_full"
inference_dataset = DATASET_NAME.ECB
inference_split = SPLIT.Test
inference_ratio = -1
inference_embed_config = EmbeddingConfig(EmbeddingEnum.ROBERTA_LARGE)

inference_model = str(LIBRARY_ROOT) + "/saved_models/ECB_ECB_310320_roberta-large_-1iter_6"

inference_event_test_file_pos = str(LIBRARY_ROOT) + "/resources/" + inference_context_set + "/" + \
                    inference_dataset.name.lower() + "/" + inference_split.name.lower() + \
                    "/Event_gold_mentions_PosPairs_Subtopic.pickle"

inference_event_test_file_neg = str(LIBRARY_ROOT) + "/resources/" + inference_context_set + \
                    "/" + inference_dataset.name.lower() + "/" + inference_split.name.lower() + \
                    "/Event_gold_mentions_NegPairs_Subtopic.pickle"

inference_embed_files = [str(LIBRARY_ROOT) + "/resources/" + inference_context_set +
                    "/" + inference_dataset.name.lower() + "/" + inference_split.name.lower() +
                    "/Event_gold_mentions_" + inference_embed_config.model_name + ".pickle"]

########################## Determenistic/Cluster System ################################
# cluster_topics = experiment without topic classification first
coref_context_set = "dataset_full"
coref_dataset = DATASET_NAME.ECB
coref_split = SPLIT.Test

coref_cluster_topics = False
coref_extractor = RelationTypeEnum.PAIRWISE
coref_cluster_type = ClusteringType.AgglomerativeClustering
coref_embed_config = EmbeddingConfig(EmbeddingEnum.BERT_LARGE_CASED)

coref_pairs_thresh = [1.0]
coref_average_link_thresh = [0.65]

coref_input_file = str(LIBRARY_ROOT) + "/resources/" + coref_context_set + "/" + coref_dataset.name.lower() + \
                "/" + coref_split.name.lower() + "/" + "Event_gold_mentions.json"

# coref_input_file = str(LIBRARY_ROOT) + "/gold_scorer/shany_ecb/ECB_Test_Event_gold_mentions_context.json"

coref_embed_util = [str(LIBRARY_ROOT) + "/resources/" + coref_context_set +
                    "/" + coref_dataset.name.lower() + "/" + coref_split.name.lower() +
                    "/Event_gold_mentions_" + coref_embed_config.model_name + ".pickle"]

coref_load_model_file = str(LIBRARY_ROOT) + "/saved_models/ECB_ECB_310320_bert-large-cased-1iter_5"
coref_scorer_out_file = str(LIBRARY_ROOT) + "/output/event_scorer_010420_bert-large" + \
                        coref_dataset.name + "_" + coref_split.name

################################################################################
# ECB_Test_Full_Event_gold_predicted_topic_mentions.json
# WEC_Dev_Full_Event_gold_mentions_validated.json
# ECB_Train_Event_gold_mentions_PosPairs_Subtopic.pickle
# ECB_Dev_Full_Event_gold_mentions_bert-large-cased.pickle
