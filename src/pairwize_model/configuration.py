from src import LIBRARY_ROOT
from src.dataobjs.dataset import DATASET_NAME

save_model_file, load_model_file, event_train_file_pos, event_train_file_neg, \
        event_validation_file_pos, event_validation_file_neg, bert_files = None, None, None, None, None, None, None


def reload():
    global save_model_file, load_model_file, event_train_file_pos, event_train_file_neg, \
        event_validation_file_pos, event_validation_file_neg, bert_files

    save_model_file = str(LIBRARY_ROOT) + "/saved_models/" + train_dataset.name + "_" + dev_dataset.name + "_final_a" + str(ratio)
    load_model_file = str(LIBRARY_ROOT) + "/final_saved_models/WEC_WEC_final_a35a3"

    event_train_file_pos = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + \
                           train_dataset.name + "_Train_Event_gold_mentions_PosPairs.pickle"
    event_train_file_neg = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + \
                           train_dataset.name + "_Train_Event_gold_mentions_NegPairs.pickle"
    event_validation_file_pos = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + \
                                dev_dataset.name + "_Dev_Event_gold_mentions_PosPairs.pickle"
    event_validation_file_neg = str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + \
                                dev_dataset.name + "_Dev_Event_gold_mentions_NegPairs.pickle"

    bert_files = [str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + train_dataset.name + "_Train_Event_gold_mentions.pickle",
                      str(LIBRARY_ROOT) + "/resources/" + context_set + "/" + dev_dataset.name + "_Dev_Event_gold_mentions.pickle"]


########################## Train Model Params ################################
train_dataset = DATASET_NAME.ECB
dev_dataset = DATASET_NAME.ECB
context_set = "dataset"

learning_rate = 1e-4
batch_size = 32
ratio = -1
iterations = 5
use_cuda = True
save_model = True
save_model_threshold = 0.1
fine_tune = False
weight_decay = 0.01
hidden_n = 150

########################## Determenistic System ################################
cluster_topics = False
dt_input_file = str(LIBRARY_ROOT) + "/resources/dataset/ECB_Test_Event_gold_mentions.json"
bert_dt_file = str(LIBRARY_ROOT) + "/resources/dataset/ECB_Test_Event_gold_mentions.pickle"
dt_load_model_file = str(LIBRARY_ROOT) + "/saved_models/ECB_ECB_final_a-1a3"
scorer_out_file = str(LIBRARY_ROOT) + "/output/ecb_test_10032020.txt"
reload()
################################################################################
