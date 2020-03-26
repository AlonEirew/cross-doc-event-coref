from src import LIBRARY_ROOT
from src.dataobjs.dataset import DATASET_NAME, SPLIT
from src.dt_system.relation_type_enum import RelationTypeEnum
from src.utils import embed_utils
from src.utils.clustering_utils import ClusteringType
from src.utils.embed_utils import BertFromFile

train_save_model_file, train_load_model_file, train_event_train_file_pos, train_event_train_file_neg, \
train_event_validation_file_pos, train_event_validation_file_neg, train_bert_files = None, None, None, None, None, None, None


def reload():
    global train_save_model_file, train_load_model_file, train_event_train_file_pos, train_event_train_file_neg, \
        train_event_validation_file_pos, train_event_validation_file_neg, train_bert_files

    train_save_model_file = str(LIBRARY_ROOT) + "/saved_models/" + train_dataset.name + "_" + dev_dataset.name + "_230320_bert_large_" + str(train_ratio)
    train_load_model_file = str(LIBRARY_ROOT) + "/saved_models/WEC_WEC_200320_bert_large_35iter_18"

    train_event_train_file_pos = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                                 train_dataset.name + "_Train_Full_Event_gold_mentions_validated_PosPairs_Subtopic.pickle"
    train_event_train_file_neg = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                                 train_dataset.name + "_Train_Full_Event_gold_mentions_validated_NegPairs_Subtopic.pickle"
    train_event_validation_file_pos = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                                      dev_dataset.name + "_Dev_Event_gold_mentions_PosPairs_Subtopic.pickle"
    train_event_validation_file_neg = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                                      dev_dataset.name + "_Dev_Event_gold_mentions_NegPairs_Subtopic.pickle"

    train_bert_files = [str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + train_dataset.name + "_Train_Full_Event_gold_mentions_validated_bert_large.pickle",
                        str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + dev_dataset.name + "_Dev_Full_Event_gold_mentions_bert_large.pickle"]


########################## Train Model Params ################################
train_dataset = DATASET_NAME.WEC
dev_dataset = DATASET_NAME.ECB
train_context_set = "dataset_full"
train_model_name = "bert-large-cased"

train_learning_rate = 1e-4
train_batch_size = 32
train_ratio = 20
train_iterations = 30
use_cuda = True
train_save_model = True
train_save_model_threshold = 0.1
train_fine_tune = False
train_weight_decay = 0.01
train_hidden_n = 150

if train_model_name == "bert-large-cased":
    train_embed_size = embed_utils.BERT_LARGE_SIZE
elif train_model_name == "bert-base-cased":
    train_embed_size = embed_utils.BERT_BASE_SIZE
elif train_model_name == "roberta.large":
    train_embed_size = embed_utils.BOBERTA_LARGE_SIZE

########################## Inference Model Params ################################
inference_dataset = DATASET_NAME.ECB
inference_split = SPLIT.Test
inference_ratio = -1
inference_context_set = "dataset_full"
inference_embed_size = embed_utils.BERT_LARGE_SIZE
inference_model = str(LIBRARY_ROOT) + "/saved_models/ECB_ECB_180320_bert_large_-1iter_5"

inference_event_test_file_pos = str(LIBRARY_ROOT) + "/resources/" + inference_context_set + "/" + \
                                inference_dataset.name + "_" + inference_split.name + "_Event_gold_mentions_PosPairs_Subtopic.pickle"
inference_event_test_file_neg = str(LIBRARY_ROOT) + "/resources/" + inference_context_set + \
                                "/" + inference_dataset.name + "_" + inference_split.name + "_Event_gold_mentions_NegPairs_Subtopic.pickle"
inference_bert = BertFromFile([str(LIBRARY_ROOT) + "/resources/" + inference_context_set +
                               "/" + inference_dataset.name + "_" + inference_split.name + "_Full_Event_gold_mentions_bert_large.pickle"],
                              inference_embed_size)

########################## Determenistic/Cluster System ################################
# cluster_topics = experiment without topic classification first
cluster_topics = False
dt_dataset = DATASET_NAME.ECB
dt_context_set = "dataset_full"
dt_extractor = RelationTypeEnum.PAIRWISE
dt_split = SPLIT.Dev
dt_cluster_type = ClusteringType.AgglomerativeClustering

dt_input_file = str(LIBRARY_ROOT) + "/resources/" + dt_context_set + "/" + dt_dataset.name + \
                "_" + dt_split.name + "_Full_Event_gold_mentions.json"
dt_bert_util = BertFromFile([str(LIBRARY_ROOT) + "/resources/" + inference_context_set +
                               "/" + dt_dataset.name + "_" + dt_split.name +
                             "_Full_Event_gold_mentions_bert_large.pickle"], embed_utils.BERT_LARGE_SIZE)


dt_load_model_file = str(LIBRARY_ROOT) + "/saved_models/ECB_ECB_180320_bert_large_-1iter_5"
scorer_out_file = str(LIBRARY_ROOT) + "/output/event_scorer_results_fine_tune_26032020"
dt_pair_thresh = [1.0]
dt_average_link_thresh = [0.6]
reload()
################################################################################
