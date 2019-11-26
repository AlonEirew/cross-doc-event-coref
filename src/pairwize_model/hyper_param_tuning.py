import logging

from src import LIBRARY_ROOT
from src.pairwize_model.train import init_basic_training_resources, train_pairwise, accuracy_on_dataset
from src.utils.dataset_utils import DATASET
from src.utils.log_utils import create_logger_with_fh

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    _use_cuda = True

    _train_dataset = [DATASET.WEC]
    _dev_dataset = [DATASET.WEC, DATASET.ECB]
    _context_set = "single_sent_clean_kenton"

    _lrs = [1e-6]
    _batch_sizes = [32]
    _alphas = [4, 6, 8, 10, 12, 14]
    _iterations = 30
    _prcs = [100]
    _use_cuda = True
    _save_model = True

    log_params_str = "hptunning_learn_train_set_ALL_dev_set_ALL"
    create_logger_with_fh(log_params_str)

    best_save_thresh = 0.1
    for tds in _train_dataset:
        for dds in _dev_dataset:
            for _batch_size in _batch_sizes:
                for _alpha in _alphas:
                    for _prc in _prcs:
                        _lr = 1e-6

                        _model_out = str(LIBRARY_ROOT) + \
                                     "/saved_models/" + tds.name + "_" + \
                                     dds.name + "_best_trained_model_a" + str(_alpha)

                        _event_train_feat, _event_validation_feat, _bert_utils, _pairwize_model = \
                            init_basic_training_resources(_context_set, tds, dds, _alpha, _use_cuda)

                        # cut_train = int((len(_event_train_feat) * _prc) / 100)
                        train_feat = _event_train_feat#_event_train_feat[0:cut_train]
                        logger.info("final train size (pos+neg)=" + str(len(train_feat)))

                        logger.info("train_set=" + tds.name + ", dev_set=" + dds.name + ", lr=" + str(_lr) + ", bs=" +
                                    str(_batch_size) + ", ratio=1:" + str(_alpha) + ", itr=" + str(_iterations) + ", percent=" + str(_prc))

                        train_pairwise(_bert_utils, _pairwize_model, train_feat,
                                                          _event_validation_feat, _batch_size,
                                                          _iterations, _lr , _use_cuda, save_model=_save_model,
                                                          model_out=_model_out, best_model_to_save=best_save_thresh)

                        # logger.info("************* EXPERIMENT TEST RESULTS **************")
                        # _pairwize_model.eval()
                        # accuracy_on_dataset("TEST", -1, _bert_utils, _pairwize_model, _event_test_feat, _use_cuda)
                        # logger.info("************* EXPERIMENT TEST RESULTS **************")

    logger.info("Process Done!")
