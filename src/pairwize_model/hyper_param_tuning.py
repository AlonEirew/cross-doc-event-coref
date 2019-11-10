import datetime

from src import LIBRARY_ROOT
from src.pairwize_model.train import run_experiment, init_basic_training_resources
from src.utils.dataset_utils import DATASET
from src.utils.log_utils import create_logger

if __name__ == '__main__':
    _use_cuda = True

    _dataset = DATASET.ECB
    _context_set = "single_sent_full_context"

    _lrs = [1e-6, 1e-7, 1e-8]
    _batch_sizes = [32, 64]
    _alphas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    _iterations = 50
    _use_cuda = True
    _save_model = False

    _event_train_file, _event_validation_file, _event_test_file, _bert_utils, _pairwize_model = \
        init_basic_training_resources(_context_set, _dataset, _use_cuda)

    for _lr in _lrs:
        for _batch_size in _batch_sizes:
            for _alpha in _alphas:
                params_str = "_lr" + str(_lr) + "_bs" + str(_batch_size) + "_a" + str(_alpha) + "_itr" + str(_iterations)
                running_timestamp = "hptuning_" + str(datetime.datetime.now().time().strftime("%H%M%S%m%d%Y"))
                log_file = str(LIBRARY_ROOT) + "/logging/" + running_timestamp + "_" + params_str + ".log"
                logger = create_logger(__name__, log_file)

                run_experiment(_event_train_file, _event_validation_file, None, _bert_utils, _pairwize_model, _batch_size, _iterations, _lr
                               , _alpha, _dataset, _use_cuda, _save_model)
