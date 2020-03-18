import json
from os import walk
from os.path import join
from xml.etree import ElementTree

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData


def read_data_from_corpus_folder(corpus):
    documents = dict()
    for (dirpath, folders, files) in walk(corpus):
        for file in files:
            is_ecb_plus = False
            if file.endswith('.xml'):
                print('processing file-', file)

                if 'ecbplus' in file:
                    is_ecb_plus = True

                tree = ElementTree.parse(join(dirpath, file))
                root = tree.getroot()
                doc_id = root.attrib['doc_name']
                sentences = dict()
                context = list()
                cur_sent = 0
                for elem in root:
                    if elem.tag == 'token':
                        sent_id = int(elem.attrib['sentence'])
                        tok_text = elem.text
                        if is_ecb_plus and sent_id == 0:
                            continue
                        if is_ecb_plus:
                            sent_id = sent_id - 1

                        if cur_sent != sent_id:
                            sentences[cur_sent] = context.copy()
                            cur_sent = sent_id
                            context.clear()

                        context.append(tok_text)

                sentences[cur_sent] = context.copy()
                documents[doc_id] = sentences

    return documents


def enhance_json_with_context(json_files, documents_context_file):

    with open(documents_context_file, 'r') as osw:
        documents_context = json.load(osw)

    for file in json_files:
        mentions = MentionData.read_mentions_json_to_mentions_data_list(file)
        for mention in mentions:
            new_mention_tokens_id = 0
            new_full_context = list()
            for sent, tokens in documents_context[mention.doc_id].items():
                if mention.sent_id == int(sent):
                    for i in range(len(mention.tokens_number)):
                        mention.tokens_number[i] += new_mention_tokens_id
                new_full_context.extend(tokens)
                new_mention_tokens_id += len(tokens)

            mention.mention_context = new_full_context

        with open(file + '_full', 'w') as osw:
            json.dump(mentions, osw, default=default, indent=4, sort_keys=True)


def default(o):
    return o.__dict__


if __name__ == '__main__':
    # print('Read all ECB+ files')
    # documents_context = read_data_from_corpus_folder(str(LIBRARY_ROOT) + '/resources/ECB+')
    # print('Done reading, star enhancing with context')
    enhance_json_with_context([str(LIBRARY_ROOT) + '/resources/dataset/ECB_Dev_Event_gold_mentions.json',
                               str(LIBRARY_ROOT) + '/resources/dataset/ECB_Test_Event_gold_mentions.json',
                               str(LIBRARY_ROOT) + '/resources/dataset/ECB_Train_Event_gold_mentions.json'],
                              str(LIBRARY_ROOT) + '/resources/dataset_full/ecb_full_contest_file.json')

    # with open(str(LIBRARY_ROOT) + '/resources/dataset_full/ecb_full_contest_file.json', 'w+') as osw:
    #     json.dump(documents_context, osw, default=default, indent=4, sort_keys=True)

    print('Done!')