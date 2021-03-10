import json
import sys

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.io_utils import write_mention_to_json
from src.utils.string_utils import StringUtils


def main(converter):
    pass
    # all_mentions = list()
    # for ment in meantime_data:
    #     all_mentions.append(converter(ment))
    #
    # write_mention_to_json(output, all_mentions)


def convert_from_meantime(mention_line):
    """
    Args:
        mention_line: a Json representation of a single mention

    Returns:
        MentionData object
    """
    try:
        mention_id = None
        topic_id = None
        coref_chain = None
        doc_id = None
        sent_id = None
        tokens_numbers = None
        mention_type = None
        predicted_coref_chain = None
        mention_context = None
        mention_index = -1

        mention_text = mention_line['tokens_str']

        if 'mention_id' in mention_line:
            mention_id = mention_line['mention_id']

        if 'topic' in mention_line:
            topic_id = mention_line['topic']

        if 'coref_chain' in mention_line:
            coref_chain = mention_line['coref_chain']

        if 'doc_id' in mention_line:
            doc_id = mention_line['doc_id']

        if 'sent_id' in mention_line:
            sent_id = mention_line['sent_id']

        if 'tokens_number' in mention_line:
            tokens_numbers = mention_line['tokens_number']

        if 'full_sentence' in mention_line:
            mention_context = mention_line['full_sentence'].split(' ')

        mention_head, mention_head_lemma, mention_ner, mention_pos = StringUtils.find_head_lemma_pos_ner(str(mention_text))

        if 'mention_type' in mention_line:
            mention_type = mention_line['mention_type']

        if 'predicted_coref_chain' in mention_line:
            predicted_coref_chain = mention_line['predicted_coref_chain']

        if 'mention_index' in mention_line:
            mention_index = mention_line['mention_index']

        mention_data = MentionData(mention_id, topic_id, doc_id, sent_id, tokens_numbers, mention_text,
                                   mention_context, mention_head, mention_head_lemma,
                                   coref_chain, mention_type, predicted_coref_chain, mention_pos, mention_ner,
                                   mention_index)
    except Exception:
        print('Unexpected error:', sys.exc_info()[0])
        raise Exception('failed reading json line-' + str(mention_line))

    return mention_data


def convert_from_gvc(mention_line):
    """
    Args:
        mention_line: a Json representation of a single mention

    Returns:
        MentionData object
    """
    try:
        mention_id = None
        topic_id = None
        coref_chain = None
        doc_id = None
        sent_id = None
        tokens_numbers = None
        predicted_coref_chain = None
        mention_context = None
        mention_index = -1

        mention_text = mention_line['MENTION_TEXT']

        if 'CLUSTER' in mention_line:
            coref_chain = mention_line['CLUSTER']

        if 'DOC' in mention_line:
            doc_id = mention_line['DOC']

        if 'SENTENCE_NUM' in mention_line:
            sent_id = mention_line['SENTENCE_NUM']

        if 'TOKEN_NUM' in mention_line:
            tokens_numbers = [mention_line['TOKEN_NUM']]

        if 'SENTENCE_TEXT' in mention_line:
            mention_context = mention_line['SENTENCE_TEXT'].split(' ')

        mention_head, mention_head_lemma, mention_pos, mention_ner = StringUtils.find_head_lemma_pos_ner(str(mention_text))

        mention_data = MentionData(mention_id, "0", doc_id, sent_id, tokens_numbers, mention_text,
                                   mention_context, mention_head, mention_head_lemma,
                                   coref_chain, "NA", predicted_coref_chain, mention_pos, mention_ner,
                                   mention_index)
    except Exception:
        print('Unexpected error:', sys.exc_info()[0])
        raise Exception('failed reading json line-' + str(mention_line))

    return mention_data


def convert_from_wikilinks(mention_line):
    """
    Args:
        mention_line: a Json representation of a single mention

    Returns:
        MentionData object
    """
    try:
        mention_id = None
        coref_chain = None
        doc_id = None
        predicted_coref_chain = None
        mention_index = -1

        mention_text = mention_line['mentionString']

        if 'corefId' in mention_line:
            coref_chain = mention_line['corefId']

        if 'extractedFromPage' in mention_line:
            doc_id = mention_line['extractedFromPage']

        mention_head, mention_head_lemma, mention_pos, mention_ner = StringUtils.find_head_lemma_pos_ner(str(mention_text))

        mention_data = MentionData(mention_id, "0", doc_id, -1, [], mention_text,
                                   [], mention_head, mention_head_lemma,
                                   coref_chain, "NA", predicted_coref_chain, mention_pos, mention_ner,
                                   mention_index)
    except Exception:
        print('Unexpected error:', sys.exc_info()[0])
        raise Exception('failed reading json line-' + str(mention_line))

    return mention_data


def main_wiki():
    clusters = json.load(open(str(LIBRARY_ROOT) + '/resources/wikilinks/wikilinks_full.json', 'r'))
    parsed_mentions = list()
    for cluster in clusters.values():
        mentions_list = cluster['mentions']
        for ment in mentions_list:
            parsed_mentions.append(convert_from_wikilinks(ment))

    output = str(LIBRARY_ROOT) + '/resources/wikilinks/Event_gold_mentions.json'
    write_mention_to_json(output, parsed_mentions)
    print("Total mentions unfiltered=" + str(len(parsed_mentions)))
    print("Done!")


if __name__ == '__main__':
    # meantime_data = json.load(open(str(LIBRARY_ROOT) + '/resources/gvc/mentions', 'r'))
    # output = str(LIBRARY_ROOT) + '/resources/gvc/Event_gold_mentions.json'
    main_wiki()
