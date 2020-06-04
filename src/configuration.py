from enum import Enum

from src import LIBRARY_ROOT
from src.coref_system.relation_extraction import RelationTypeEnum
from src.dataobjs.dataset import Split, EcbDataSet, WecDataSet, DataSet
from src.utils.clustering_utils import ClusteringType
from src.utils.embed_utils import EmbeddingConfig, EmbeddingEnum

########################## Train Model Params ################################


class ConfigType(Enum):
    Train = 1
    Inference = 2
    Clustering = 3


class Configuration(object):
    def __init__(self, config_type):
        self.learning_rate = -1
        self.batch_size = -1
        self.ratio = -1
        self.iterations = -1
        self.use_cuda = True
        self.save_model = True
        self.save_model_threshold = -1
        self.fine_tune = False
        self.weight_decay = -1
        self.hidden_n = 150
        self.context_set = "dataset_full"

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        self.embed_config = None
        self.save_model_file = None
        self.load_model_file = None

        self.event_train_file_pos = None
        self.event_train_file_neg = None
        self.event_validation_file_pos = None
        self.event_validation_file_neg = None
        self.event_test_file_pos = None
        self.event_test_file_neg = None

        self.embed_files = None
        self.split = None
        self.mentions_file = None

        self.cluster_extractor = None
        self.to_single_topic = False
        self.cluster_algo_type = None
        self.cluster_pairs_thresh = None
        self.cluster_average_link_thresh = None

        if config_type == ConfigType.Train:
            self.init_train()
        elif config_type == ConfigType.Inference:
            self.init_inference()
        elif config_type == ConfigType.Clustering:
            self.init_clustering()

    def init_train(self):
        """ TRAIN CONFIGURATION """
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.ratio = 1
        self.iterations = 10
        self.use_cuda = True
        self.save_model = True
        self.save_model_threshold = 0.01
        self.fine_tune = False
        self.weight_decay = 0.01
        self.hidden_n = 150

        self.context_set = "dataset_full"
        self.train_dataset = WecDataSet(ratio=self.ratio, split=Split.Train)
        self.dev_dataset = EcbDataSet() #WecDataSet(split=Split.Dev)
        self.embed_config = EmbeddingConfig(EmbeddingEnum.ROBERTA_LARGE)

        self.save_model_file = str(LIBRARY_ROOT) + "/saved_models/" + self.train_dataset.name + "_" + self.dev_dataset.name + \
                                "_310520_" + self.embed_config.embed_type.name.lower() + "_" + str(self.ratio)

        self.load_model_file = str(LIBRARY_ROOT) + "/saved_models/WEC_WEC_200320_bert_large_35iter_18"

        self.event_train_file_pos = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + \
                                     self.train_dataset.name.lower() + "/train/Event_gold_mentions_validated4_PosPairs.pickle"
        self.event_train_file_neg = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + \
                                     self.train_dataset.name.lower() + "/train/Event_gold_mentions_validated4_NegPairs.pickle"
        self.event_validation_file_pos = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + \
                                          self.dev_dataset.name.lower() + "/dev/Event_gold_mentions_validated2_PosPairs.pickle"
        self.event_validation_file_neg = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + \
                                          self.dev_dataset.name.lower() + "/dev/Event_gold_mentions_validated2_NegPairs.pickle"

        self.embed_files = [str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + self.train_dataset.name.lower() +
                             "/train/Event_gold_mentions_validated4_" + self.embed_config.embed_type.name.lower() + ".pickle",
                             str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + self.dev_dataset.name.lower() +
                             "/dev/Event_gold_mentions_validated2_" + self.embed_config.embed_type.name.lower() + ".pickle",
                            ]

    def init_inference(self):
        """ INFERENCE CONFIGURATION """
        self.context_set = "dataset_full"
        self.split = Split.Test
        self.ratio = -1
        self.test_dataset = EcbDataSet() #WecDataSet(ratio=self.ratio, split=self.split)
        self.embed_config = EmbeddingConfig(EmbeddingEnum.ROBERTA_LARGE)

        self.load_model_file = str(LIBRARY_ROOT) + "/saved_models/WEC_ECB_120420_max_roberta_large_30iter_8"

        self.event_test_file_pos = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + \
                            self.test_dataset.name.lower() + "/" + self.split.name.lower() + \
                            "/Event_gold_mentions_PosPairs_Subtopic.pickle"

        self.event_test_file_neg = str(LIBRARY_ROOT) + "/resources/" + self.context_set + \
                            "/" + self.test_dataset.name.lower() + "/" + self.split.name.lower() + \
                            "/Event_gold_mentions_NegPairs_Subtopic.pickle"

        self.embed_files = [str(LIBRARY_ROOT) + "/resources/" + self.context_set +
                            "/" + self.test_dataset.name.lower() + "/" + self.split.name.lower() +
                            "/Event_gold_mentions_" + self.embed_config.embed_type.name.lower() + ".pickle"]

    def init_clustering(self):
        """ COREF CONFIGURATION """
        # cluster_topics = experiment without topic classification first
        self.context_set = "dataset_full"
        self.split = Split.Dev
        self.test_dataset = WecDataSet()

        self.to_single_topic = True
        self.cluster_extractor = RelationTypeEnum.SAME_HEAD_LEMMA
        self.cluster_algo_type = ClusteringType.NaiveClustering
        self.embed_config = EmbeddingConfig(EmbeddingEnum.ROBERTA_LARGE)

        self.cluster_pairs_thresh = [1.0]
        self.cluster_average_link_thresh = [1.0]

        self.mentions_file = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + self.test_dataset.name.lower() + \
                        "/" + self.split.name.lower() + "/" + "Event_gold_mentions_clean11.json"

        self.embed_files = [str(LIBRARY_ROOT) + "/resources/" + self.context_set +
                            "/" + self.test_dataset.name.lower() + "/" + self.split.name.lower() +
                            "/Event_gold_mentions_" + self.embed_config.embed_type.name.lower() + ".pickle"]

        self.load_model_file = str(LIBRARY_ROOT) + "/saved_models/ECB_ECB_020520_EcbWecDev10_roberta_large_-1iter_4"
        self.save_model_file = str(LIBRARY_ROOT) + "/output/event_scorer_lemma_" + \
                                self.test_dataset.name + "_" + self.split.name

################################################################################
# ECB_Test_Full_Event_gold_predicted_topic_mentions.json
# WEC_Dev_Full_Event_gold_mentions_validated2.json
# ECB_Train_Event_gold_mentions_PosPairs_Subtopic.pickle
# ECB_Dev_Full_Event_gold_mentions_bert-large-cased.pickle
