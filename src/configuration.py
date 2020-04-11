from enum import Enum

from src import LIBRARY_ROOT
from src.coref_system.relation_extraction import RelationTypeEnum
from src.dataobjs.dataset import Split, EcbDataSet, WecDataSet
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
        self.cluster_topics = False
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
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.ratio = -1
        self.iterations = 10
        self.use_cuda = True
        self.save_model = True
        self.save_model_threshold = 0.01
        self.fine_tune = False
        self.weight_decay = 0.01
        self.hidden_n = 150

        self.context_set = "dataset_full"
        self.train_dataset = EcbDataSet() #WecDataSet(ratio=train_ratio, split=Split.Train)
        self.dev_dataset = EcbDataSet() #WecDataSet(split=Split.Dev)
        self.embed_config = EmbeddingConfig(EmbeddingEnum.ROBERTA_LARGE)

        self.save_model_file = str(LIBRARY_ROOT) + "/saved_models/" + self.train_dataset.name + "_" + self.dev_dataset.name + \
                                "_100420_11_" + self.embed_config.embed_type.name.lower() + "_" + str(self.ratio)

        self.load_model_file = str(LIBRARY_ROOT) + "/saved_models/WEC_WEC_200320_bert_large_35iter_18"

        self.event_train_file_pos = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + \
                                     self.train_dataset.name.lower() + "/train/Event_gold_mentions_PosPairs_Subtopic.pickle"
        self.event_train_file_neg = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + \
                                     self.train_dataset.name.lower() + "/train/Event_gold_mentions_NegPairs_Subtopic.pickle"
        self.event_validation_file_pos = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + \
                                          self.dev_dataset.name.lower() + "/dev/Event_gold_mentions_PosPairs_Subtopic.pickle"
        self.event_validation_file_neg = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + \
                                          self.dev_dataset.name.lower() + "/dev/Event_gold_mentions_NegPairs_Subtopic.pickle"

        self.embed_files = [str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + self.train_dataset.name.lower() +
                             "/train/Event_gold_mentions_" + self.embed_config.embed_type.name.lower() + ".pickle",
                             str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + self.dev_dataset.name.lower() +
                             "/dev/Event_gold_mentions_" + self.embed_config.embed_type.name.lower() + ".pickle"]

    def init_inference(self):
        self.context_set = "dataset_full"
        self.split = Split.Dev
        self.ratio = -1
        self.test_dataset = WecDataSet(ratio=self.ratio, split=self.split)
        self.embed_config = EmbeddingConfig(EmbeddingEnum.ROBERTA_LARGE)

        self.load_model_file = str(LIBRARY_ROOT) + "/saved_models/WEC_WEC_010420_roberta-large_30iter_4"

        self.event_test_file_pos = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + \
                            self.test_dataset.name.lower() + "/" + self.split.name.lower() + \
                            "/Event_gold_mentions_validated_PosPairs_Subtopic.pickle"

        self.event_test_file_neg = str(LIBRARY_ROOT) + "/resources/" + self.context_set + \
                            "/" + self.test_dataset.name.lower() + "/" + self.split.name.lower() + \
                            "/Event_gold_mentions_validated_NegPairs_Subtopic.pickle"

        self.embed_files = [str(LIBRARY_ROOT) + "/resources/" + self.context_set +
                            "/" + self.test_dataset.name.lower() + "/" + self.split.name.lower() +
                            "/Event_gold_mentions_validated_" + self.embed_config.model_name + ".pickle"]

    def init_clustering(self):
        # cluster_topics = experiment without topic classification first
        self.context_set = "dataset_full"
        self.split = Split.Test
        self.test_dataset = EcbDataSet() #WecDataSet(-1, self.split)

        self.cluster_topics = False
        self.cluster_extractor = RelationTypeEnum.PAIRWISE
        self.cluster_algo_type = ClusteringType.AgglomerativeClustering
        self.embed_config = EmbeddingConfig(EmbeddingEnum.ROBERTA_LARGE)

        self.cluster_pairs_thresh = [1.0]
        self.cluster_average_link_thresh = [0.65]

        self.mentions_file = str(LIBRARY_ROOT) + "/resources/" + self.context_set + "/" + self.test_dataset.name.lower() + \
                        "/" + self.split.name.lower() + "/" + "Event_pred_mentions.json"

        self.embed_files = [str(LIBRARY_ROOT) + "/resources/" + self.context_set +
                            "/" + self.test_dataset.name.lower() + "/" + self.split.name.lower() +
                            "/Event_gold_mentions_" + self.embed_config.embed_type.name.lower() + ".pickle"]

        self.load_model_file = str(LIBRARY_ROOT) + "/saved_models/ECB_ECB_100420_11_roberta_large_-1iter_5"
        self.save_model_file = str(LIBRARY_ROOT) + "/output/event_scorer_110420_11_pred_" + \
                                self.test_dataset.name + "_" + self.split.name

################################################################################
# ECB_Test_Full_Event_gold_predicted_topic_mentions.json
# WEC_Dev_Full_Event_gold_mentions_validated2.json
# ECB_Train_Event_gold_mentions_PosPairs_Subtopic.pickle
# ECB_Dev_Full_Event_gold_mentions_bert-large-cased.pickle
