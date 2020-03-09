import logging
from typing import List

from src.dataobjs.mention_data import MentionData
from src.utils.json_utils import load_mentions_from_json_file

logger = logging.getLogger(__name__)


class Topic(object):
    def __init__(self, topic_id):
        self.topic_id = topic_id
        self.mentions = []


class Topics(object):
    def __init__(self):
        self.topics_list = []
        self.keep_order = False

    def topic_id_exists(self, id_to_search):
        for topic in self.topics_list:
            if topic.topic_id == id_to_search:
                return True

        return False

    def get_topic_by_id(self, id_to_search):
        for topic in self.topics_list:
            if topic.topic_id == id_to_search:
                return topic

        return None

    def create_from_file(self, mentions_file_path: str, keep_order: bool = False) -> None:
        """

        Args:
            keep_order: whether to keep original mentions order or not (default = False)
            mentions_file_path: this topic mentions json file
        """
        self.keep_order = keep_order
        mentions = load_mentions_from_json_file(mentions_file_path)
        self.topics_list = self.order_mentions_by_topics(mentions)

    def order_mentions_by_topics(self, mentions: str) -> List[Topic]:
        """
        Order mentions to documents topics
        Args:
            mentions: json mentions file

        Returns:
            List[Topic] of the mentions separated by their documents topics
        """
        running_index = 0
        topics = []
        current_topic_ref = None
        for mention_line in mentions:
            mention = MentionData.read_json_mention_data_line(mention_line)

            if self.keep_order:
                if mention.mention_index == -1:
                    mention.mention_index = running_index
                    running_index += 1

            topic_id = mention.topic_id

            if not current_topic_ref or len(topics) > 0 and topic_id != topics[-1].topic_id:
                current_topic_ref = Topic(topic_id)
                topics.append(current_topic_ref)

            current_topic_ref.mentions.append(mention)

        return topics

    def to_single_topic(self):
        new_topic = Topic(-1)

        for topic in self.topics_list:
            for ment in topic.mentions:
                new_topic.mentions.append(ment)

        self.topics_list.clear()
        self.topics_list.append(new_topic)

    def convert_to_clusters(self):
        clusters = dict()
        for topic in self.topics_list:
            for mention in topic.mentions:
                if mention.coref_chain not in clusters:
                    clusters[mention.coref_chain] = list()
                clusters[mention.coref_chain].append(mention)
            # break
        return clusters
