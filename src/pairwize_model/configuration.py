from src import LIBRARY_ROOT
from src.utils.dataset_utils import DATASET


save_model_file, load_model_file, event_train_file_pos, event_train_file_neg, \
        event_validation_file_pos, event_validation_file_neg, bert_files = None, None, None, None, None, None, None


def reload():
    global save_model_file, load_model_file, event_train_file_pos, event_train_file_neg, \
        event_validation_file_pos, event_validation_file_neg, bert_files

    save_model_file = str(LIBRARY_ROOT) + "/saved_models/" + train_dataset.name + "_" + dev_dataset.name + "_final_a" + str(ratio)
    load_model_file = str(LIBRARY_ROOT) + "/saved_models/WEC_trained_model_1"

    event_train_file_pos = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + \
                           train_dataset.name + "_Train_Event_gold_mentions_PosPairs_NoTopicClust.pickle"
    event_train_file_neg = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + \
                           train_dataset.name + "_Train_Event_gold_mentions_NegPairs_NoTopicClust.pickle"
    event_validation_file_pos = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + \
                                dev_dataset.name + "_Dev_Event_gold_mentions_PosPairs.pickle"
    event_validation_file_neg = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + \
                                dev_dataset.name + "_Dev_Event_gold_mentions_NegPairs.pickle"

    bert_files = [str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + train_dataset.name + "_Train_Event_gold_mentions.pickle",
                      str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + dev_dataset.name + "_Dev_Event_gold_mentions.pickle"]


########################## Train Model Params ################################
train_dataset = DATASET.ECB
dev_dataset = DATASET.WEC
context_set = "final_dataset"

learning_rate = 1e-5
batch_size = 32
ratio = 80
iterations = 1
use_cuda = True
save_model = True
save_model_threshold = 0.1
fine_tune = False
weight_decay = 0.01
hidden_n = 150

########################## Determenistic System ################################
cluster_topics = False
dt_input_file = str(LIBRARY_ROOT) + "/resources/final_dataset/WEC_Dev_Event_gold_mentions.json"
bert_dt_file = str(LIBRARY_ROOT) + "/resources/final_dataset/WEC_Dev_Event_gold_mentions.pickle"
scorer_out_file = str(LIBRARY_ROOT) + "/output/event_scorer_results_lemma_wec_dev_full.txt"
reload()
################################################################################
