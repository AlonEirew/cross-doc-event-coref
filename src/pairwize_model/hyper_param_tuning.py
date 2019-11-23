import logging

from src import LIBRARY_ROOT
from src.pairwize_model.train import init_basic_training_resources, train_pairwise, accuracy_on_dataset
from src.utils.dataset_utils import DATASET
from src.utils.log_utils import create_logger_with_fh

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    _use_cuda = True

    _train_dataset = DATASET.WEC
    _dev_dataset = DATASET.WEC
    _context_set = "single_sent_clean_mean"

    _lrs = [1e-7]
    _batch_sizes = [32]
    _alphas = [10]
    _iterations = 30
    _prcs = [20, 40, 60, 80]
    _use_cuda = True
    _save_model = False
    _model_out = str(LIBRARY_ROOT) + "/saved_models/" + _train_dataset.name + "_" + _dev_dataset.name +"_best_trained_model"

    log_params_str = "hptunning_learn_curve_train_set_" + _train_dataset.name + "_test_set_" + _dev_dataset.name
    create_logger_with_fh(log_params_str)

    best_save_thresh = 0.2
    for _lr in _lrs:
        for _batch_size in _batch_sizes:
            for _alpha in _alphas:
                for _prc in _prcs:
                    _event_train_feat, _event_validation_feat, _event_test_feat, _bert_utils, _pairwize_model = \
                        init_basic_training_resources(_context_set, _train_dataset, _dev_dataset, _alpha, _use_cuda)

                    cut_train = int((len(_event_train_feat) * _prc) / 100)
                    train_feat = _event_train_feat[0:cut_train]
                    logger.info("final train size (pos+neg)=" + str(len(train_feat)))

                    logger.info("train_set=" + _train_dataset.name + ", dev_set" + _dev_dataset.name + ", lr=" + str(_lr) + ", bs=" +
                                str(_batch_size) + ", ratio=1:" + str(_alpha) + ", itr=" + str(_iterations) + ", percent=" + str(_prc))

                    train_pairwise(_bert_utils, _pairwize_model, train_feat,
                                                      _event_validation_feat, _batch_size,
                                                      _iterations, _lr , _use_cuda, save_model=_save_model,
                                                      model_out=_model_out, best_model_to_save=best_save_thresh)

                    logger.info("************* EXPERIMENT TEST RESULTS **************")
                    _pairwize_model.eval()
                    accuracy_on_dataset("TEST", -1, _bert_utils, _pairwize_model, _event_test_feat, _use_cuda)
                    logger.info("************* EXPERIMENT TEST RESULTS **************")

    logger.info("Process Done!")
