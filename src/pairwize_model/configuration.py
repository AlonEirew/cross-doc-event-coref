from src import LIBRARY_ROOT
from src.dataobjs.dataset import DATASET_NAME, SPLIT
from src.utils.embed_utils import BertFromFile

train_save_model_file, train_load_model_file, train_event_train_file_pos, train_event_train_file_neg, \
train_event_validation_file_pos, train_event_validation_file_neg, train_bert_files = None, None, None, None, None, None, None


def reload():
    global train_save_model_file, train_load_model_file, train_event_train_file_pos, train_event_train_file_neg, \
        train_event_validation_file_pos, train_event_validation_file_neg, train_bert_files

    train_save_model_file = str(LIBRARY_ROOT) + "/saved_models/" + train_dataset.name + "_" + dev_dataset.name + "_140320_2_" + str(train_ratio)
    train_load_model_file = str(LIBRARY_ROOT) + "/final_saved_models/WEC_WEC_final_a35a3"

    train_event_train_file_pos = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                                 train_dataset.name + "_Train_Event_gold_mentions_PosPairs_Subtopic.pickle"
    train_event_train_file_neg = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                                 train_dataset.name + "_Train_Event_gold_mentions_NegPairs_Subtopic.pickle"
    train_event_validation_file_pos = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                                      dev_dataset.name + "_Dev_Event_gold_mentions_PosPairs_Subtopic.pickle"
    train_event_validation_file_neg = str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + \
                                      dev_dataset.name + "_Dev_Event_gold_mentions_NegPairs_Subtopic.pickle"

    train_bert_files = [str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + train_dataset.name + "_Train_Event_gold_mentions_bert.pickle",
                        str(LIBRARY_ROOT) + "/resources/" + train_context_set + "/" + dev_dataset.name + "_Dev_Event_gold_mentions_bert.pickle"]


########################## Train Model Params ################################
train_dataset = DATASET_NAME.ECB
dev_dataset = DATASET_NAME.ECB
train_context_set = "dataset"

train_learning_rate = 1e-4
train_batch_size = 32
train_ratio = -1
train_iterations = 30
use_cuda = True
train_save_model = True
train_save_model_threshold = 0.1
train_fine_tune = False
train_weight_decay = 0.01
train_hidden_n = 150

########################## Inference Model Params ################################
inference_dataset = DATASET_NAME.ECB
inference_split = SPLIT.Test
inference_ratio = -1
inference_context_set = "dataset_full"
inference_model = str(LIBRARY_ROOT) + "/saved_models/ECB_ECB_140320_1-1iter_5"

inference_event_test_file_pos = str(LIBRARY_ROOT) + "/resources/" + inference_context_set + "/" + \
                                inference_dataset.name + "_" + inference_split.name + "_Event_gold_mentions_PosPairs_Subtopic.pickle"
inference_event_test_file_neg = str(LIBRARY_ROOT) + "/resources/" + inference_context_set + \
                                "/" + inference_dataset.name + "_" + inference_split.name + "_Event_gold_mentions_NegPairs_Subtopic.pickle"
inference_bert = BertFromFile([str(LIBRARY_ROOT) + "/resources/" + inference_context_set +
                               "/" + inference_dataset.name + "_" + inference_split.name + "_Full_Event_gold_mentions_bert.pickle"])

########################## Determenistic/Cluster System ################################
cluster_topics = False
dt_dataset = DATASET_NAME.ECB
dt_context_set = "dataset_full"

dt_input_file = str(LIBRARY_ROOT) + "/resources/" + dt_context_set + "/" + dt_dataset.name + \
                "_Test_Full_Event_gold_mentions.json"
dt_bert_file = str(LIBRARY_ROOT) + "/resources/" + dt_context_set + "/" + dt_dataset.name + \
               "_Test_Full_Event_gold_mentions_bert.pickle"

dt_load_model_file = str(LIBRARY_ROOT) + "/saved_models/ECB_ECB_140320_1-1iter_5"
scorer_out_file = str(LIBRARY_ROOT) + "/output/event_scorer_results_ecb_test_15032020.txt"
dt_pair_thresh = 0.3
dt_average_link_thresh = 1.0
reload()
################################################################################
