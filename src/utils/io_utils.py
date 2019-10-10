import json

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_json_file(file_path):
    """load a file into a json object"""
    try:
        with open(file_path) as small_file:
            return json.load(small_file)
    except OSError as e:
        print(e)
        print('trying to read file in blocks')
        with open(file_path) as big_file:
            json_string = ''
            while True:
                block = big_file.read(64 * (1 << 20))  # Read 64 MB at a time;
                json_string = json_string + block
                if not block:  # Reached EOF
                    break
            return json.loads(json_string)


def write_coref_scorer_results(mentions, output_file: str):
    """
    :param mentions: List[MentionData]
    :param output_file: str
    :return:
    """
    output = open(output_file, 'w')
    output.write('#begin document (ECB+/ecbplus_all); part 000\n')
    for mention in mentions:
        output.write('ECB+/ecbplus_all\t' + '(' + str(mention.predicted_coref_chain) + ')\n')
    output.write('#end document')
    output.close()
