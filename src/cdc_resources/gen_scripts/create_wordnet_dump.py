import argparse
import json
import logging

from nlp_architect.common.cdc.mention_data import MentionData
from nlp_architect.data.cdc_resources.wordnet.wordnet_online import WordnetOnline
from nlp_architect.utils import io
from nlp_architect.utils.io import json_dumper

from src.utils.io_utils import read_mentions_json_to_mentions_data_list

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Create WordNet dataset only dump')

parser.add_argument('--mentions', type=str, help='mentions file', required=True)

parser.add_argument('--output', type=str, help='location were to create dump file', required=True)

args = parser.parse_args()


def wordnet_dump():
    out_file = args.output
    mentions_file = args.mentions
    logger.info('Loading mentions files...')
    mentions = read_mentions_json_to_mentions_data_list(mentions_file)
    logger.info('Done loading mentions files, starting local dump creation...')
    result_dump = dict()
    wordnet = WordnetOnline()
    for mention in mentions:
        page = wordnet.get_pages(mention)
        result_dump[page.orig_phrase] = page

    with open(out_file, 'w') as out:
        json.dump(result_dump, out, default=json_dumper)

    logger.info('Wordnet Dump Created Successfully, '
                'extracted total of %d wn pages', len(result_dump))
    logger.info('Saving dump to file-%s', out_file)


if __name__ == '__main__':
    io.validate_existing_filepath(args.mentions)
    io.validate_existing_filepath(args.output)
    wordnet_dump()
