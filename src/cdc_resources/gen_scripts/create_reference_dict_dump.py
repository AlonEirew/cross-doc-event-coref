import argparse
import json
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Create Referent dictionary dataset only dump')

parser.add_argument('--ref_dict', type=str, help='referent dictionary file', required=True)

parser.add_argument('--mentions', type=str, help='dataset mentions', required=True)

parser.add_argument('--output', type=str, help='location were to create dump file', required=True)

args = parser.parse_args()


def ref_dict_dump():
    logger.info('Extracting referent dict dump, this may take a while...')
    ref_dict_file = args.ref_dict
    out_file = args.output
    mentions_entity_gold_file = [args.mentions]
    vocab = load_mentions_vocab_from_files(mentions_entity_gold_file, True)

    ref_dict = ReferentDictRelationExtraction.load_reference_dict(ref_dict_file)

    ref_dict_for_vocab = {}
    for word in vocab:
        if word in ref_dict:
            ref_dict_for_vocab[word] = ref_dict[word]

    logger.info('Found %d words from vocabulary', len(ref_dict_for_vocab.keys()))
    logger.info('Preparing to save refDict output file')
    with open(out_file, 'w') as f:
        json.dump(ref_dict_for_vocab, f)
    logger.info('Done saved to-%s', out_file)


if __name__ == '__main__':
    io.validate_existing_filepath(args.mentions)
    io.validate_existing_filepath(args.output)
    io.validate_existing_filepath(args.ref_dict)
    ref_dict_dump()
