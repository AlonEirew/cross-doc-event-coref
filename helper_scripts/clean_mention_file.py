import re

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.io_utils import write_mention_to_json
from src.utils.string_utils import StringUtils


def clean_bad_mentions():
    total_bad_cntx = 0
    total_years = 0
    final_mentions = list()
    for mention in mentions:
        mention.tokens_str = mention.tokens_str.strip()
        if is_bad_context(mention):
            total_bad_cntx += 1
        elif is_year(mention):
            total_years += 1
        else:
            final_mentions.append(mention)

    print(str(total_bad_cntx))
    print(str(total_years))
    print(str(len(final_mentions)))
    write_mention_to_json(
        str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_clean2.json', final_mentions)


def is_year(mention):
    if re.match(r'^\b\d{1,4}\b$', mention.tokens_str):
        return True
    return False


def is_bad_context(mention):
    for i, tok_id in enumerate(mention.tokens_number):
        mention_text = list(list(zip(*StringUtils.get_tokenized_string(mention.tokens_str)[0]))[0])
        try:
            if mention_text[i] != mention.mention_context[tok_id]:
                return True
        except:
            return True
    return False


if __name__ == '__main__':
    mentions = MentionData.read_mentions_json_to_mentions_data_list(
        str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_clean1.json')
    print("total mention at start=" + str(len(mentions)))
    clean_bad_mentions()
