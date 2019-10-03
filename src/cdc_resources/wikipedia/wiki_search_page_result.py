from nlp_architect.data.cdc_resources.data_types.wiki.wikipedia_page import WikipediaPage


class WikipediaSearchPageResult(object):
    def __init__(self, search_phrase: str, page_result: WikipediaPage):
        """
        Args:
            search_phrase: the search phrase that yield this page result
            page_result: page result for search phrase
        """
        self.search_phrase = search_phrase
        self.page_result = page_result
