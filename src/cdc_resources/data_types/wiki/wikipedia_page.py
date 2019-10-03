import re
from typing import Dict

from src.cdc_resources.data_types.wiki.wikipedia_page_extracted_relations import WikipediaPageExtractedRelations, \
    DISAMBIGUATION_TITLE
from src.utils.string_utils import StringUtils


class WikipediaPage(object):
    def __init__(self, orig_phrase: str = None, orig_phrase_norm: str = None,
                 wiki_title: str = None, wiki_title_norm: str = None,
                 score: int = 0, pageid: int = 0, description: str = None,
                 relations: WikipediaPageExtractedRelations = None) -> None:
        """
        Object represent a Wikipedia Page and extracted fields.

        Args:
            orig_phrase (str): original search phrase
            orig_phrase_norm (str): original search phrase normalized
            wiki_title (str): page title
            wiki_title_norm (str): page title normalized
            score (int): score for getting wiki_title from orig_phrase
            pageid (int): the unique page identifier
            description (str, optional): the page description
            relations (WikipediaPageExtractedRelations): Object that represent all
                                                         extracted Wikipedia relations
        """
        self.orig_phrase = orig_phrase
        if orig_phrase_norm is None:
            self.orig_phrase_norm = StringUtils.normalize_str(orig_phrase)
        else:
            self.orig_phrase_norm = orig_phrase_norm

        self.wiki_title = wiki_title.replace(DISAMBIGUATION_TITLE, '')
        if wiki_title_norm is None:
            self.wiki_title_norm = StringUtils.normalize_str(wiki_title)
        else:
            self.wiki_title_norm = wiki_title_norm

        self.score = score
        self.pageid = int(pageid)
        self.description = description
        self.relations = relations

    def toJson(self) -> Dict:
        result_dict = {}
        result_dict['orig_phrase'] = self.orig_phrase
        result_dict['orig_phrase_norm'] = self.orig_phrase_norm
        result_dict['wiki_title'] = self.wiki_title
        result_dict['wiki_title_norm'] = self.wiki_title_norm
        result_dict['score'] = self.score
        result_dict['pageid'] = self.pageid
        result_dict['description'] = self.description
        result_dict['relations'] = self.relations.toJson()
        return result_dict

    def __eq__(self, other):
        return self.orig_phrase == other.orig_phrase and self.wiki_title == other.wiki_title and \
            self.pageid == other.pageid

    def __hash__(self):
        return hash(self.orig_phrase) + hash(self.pageid) + hash(self.wiki_title)

    def __str__(self) -> str:
        result_str = ''
        try:
            title_strip = re.sub(u'(\u2018|\u2019)', '\'', self.orig_phrase)
            wiki_title_strip = re.sub(u'(\u2018|\u2019)', '\'', self.wiki_title)
            result_str = str(title_strip) + ', ' + str(wiki_title_strip) + ', ' + \
                str(self.score) + ', ' + str(self.pageid) + ', ' + \
                str(self.description) + ', ' + str(self.relations)
        except Exception:
            result_str = 'error in to_string()'

        return result_str
