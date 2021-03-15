"""
Usage:
    inference.py --tpf=<TestPosFile> --tnf=<testNegFile> --te=<TestEmbed> --mf=<ModelFile> [--cuda=<b>]

Options:
    -h --help       Show this screen.
    --cuda=<y>      True/False - Whether to use cuda device or not [default: True]

"""

import logging
import ntpath

import torch
from docopt import docopt
from src.train import accuracy_on_dataset
from src.utils.log_utils import create_logger_with_fh
from src.dataobjs.dataset import EcbDataSet
from src.utils.embed_utils import EmbedFromFile

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    _arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    print(_arguments)
    _dataset_arg = _arguments.get("--dataset")
    _model_in = _arguments.get("--mf")
    _event_test_file_pos = _arguments.get("--tpf")
    _event_test_file_neg = _arguments.get("--tnf")
    _embed_file = _arguments.get("--te")
    _use_cuda = True if _arguments.get("--cuda").lower() == "true" else False

    _dataset = EcbDataSet()

    log_param_str = "inference_" + ntpath.basename(_model_in) + ".log"
    create_logger_with_fh(log_param_str)

    logger.info("Loading the model from-" + _model_in)
    _pairwize_model = torch.load(_model_in)
    _embed_utils = EmbedFromFile([_embed_file])
    _pairwize_model.set_embed_utils(_embed_utils)
    _pairwize_model.eval()

    positive_ = _dataset.load_pos_pickle(_event_test_file_pos)
    negative_ = _dataset.load_neg_pickle(_event_test_file_neg)
    split_feat = _dataset.create_features_from_pos_neg(positive_, negative_)

    accuracy_on_dataset("", 0, _pairwize_model, split_feat)
