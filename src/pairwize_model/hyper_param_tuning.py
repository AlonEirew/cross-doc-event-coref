import logging

from src import LIBRARY_ROOT, configuration
from src.dataobjs.dataset import WecDataSet, EcbDataSet
from src.pairwize_model.train import init_basic_training_resources, train_pairwise, accuracy_on_dataset
from src.utils.log_utils import create_logger_with_fh

logger = logging.getLogger(__name__)


def set_experiment_configuration(context_set, tds, dds, ration):
    configuration.train_context_set = context_set
    configuration.train_dataset = tds
    configuration.dev_dataset = dds
    configuration.train_ratio = ration


if __name__ == '__main__':
    _use_cuda = True

    _train_dataset = [WecDataSet()]
    _dev_dataset = [EcbDataSet()]
    _context_set = "final_dataset"

    _lrs = [1e-5]
    _batch_sizes = [32]
    _alphas = [25]
    _iterations = 5
    _prcs = [50]
    _save_model = True

    log_params_str = "hptunning_train_set_" + _train_dataset[0].name + "_dev_set_" + _dev_dataset[0].name + "_validated"
    create_logger_with_fh(log_params_str)

    best_save_thresh = 0.1
    for _tds in _train_dataset:
        for _dds in _dev_dataset:
            for _batch_size in _batch_sizes:
                for _alpha in _alphas:
                    for _prc in _prcs:
                        for _lr in _lrs:
                            _model_out = str(LIBRARY_ROOT) + \
                                         "/saved_models/" + _tds.name + "_" + \
                                         _dds.name + "_hptuned_p" + str(_prc)

                            set_experiment_configuration(_context_set, _tds, _dds, _alpha)

                            _event_train_feat, _event_validation_feat, _pairwize_model = init_basic_training_resources()

                            cut_train = int((len(_event_train_feat) * _prc) / 100)
                            train_feat = _event_train_feat[0:cut_train]
                            # train_feat = _event_train_feat
                            logger.info("final train size (pos+neg)=" + str(len(train_feat)))

                            logger.info(
                                "train_set=" + configuration.train_dataset.name + ", dev_set=" + configuration.dev_dataset.name +
                                ", lr=" + str(configuration.train_learning_rate) + ", bs=" + str(configuration.train_batch_size) +
                                ", ratio=1:" + str(configuration.train_ratio) + ", itr=" + str(configuration.train_iterations) +
                                ", hidden_n=" + str(configuration.train_hidden_n) + ", weight_decay=" + str(
                                    configuration.train_weight_decay))

                            train_pairwise(_pairwize_model, train_feat,
                                                              _event_validation_feat, _batch_size,
                                                              _iterations, _lr, save_model=_save_model,
                                                              model_out=_model_out, best_model_to_save=best_save_thresh)

                            # logger.info("************* EXPERIMENT TEST RESULTS **************")
                            # _pairwize_model.eval()
                            # accuracy_on_dataset("TEST", -1, _embed_utils, _pairwize_model, _event_test_feat, _use_cuda)
                            # logger.info("************* EXPERIMENT TEST RESULTS **************")

    logger.info("Process Done!")
