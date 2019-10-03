from typing import List

from src.obj.mention_data import MentionData


class Cluster(object):
    def __init__(self, coref_chain: int = -1) -> None:
        """
        Object represent a set of mentions with same coref chain id

        Args:
            coref_chain (int): the cluster id/coref_chain value
        """
        self.mentions = []
        self.cluster_strings = []
        self.merged = False
        self.coref_chain = coref_chain
        self.mentions_corefs = set()

    def get_mentions(self):
        return self.mentions

    def add_mention(self, mention: MentionData) -> None:
        if mention is not None:
            mention.predicted_coref_chain = self.coref_chain
            self.mentions.append(mention)
            self.cluster_strings.append(mention.tokens_str)
            self.mentions_corefs.add(mention.coref_chain)

    def merge_clusters(self, cluster) -> None:
        """
        Args:
            cluster: cluster to merge this cluster with
        """
        for mention in cluster.mentions:
            mention.predicted_coref_chain = self.coref_chain

        self.mentions.extend(cluster.mentions)
        self.cluster_strings.extend(cluster.cluster_strings)
        self.mentions_corefs.update(cluster.mentions_corefs)

    def get_cluster_id(self) -> str:
        """
        Returns:
            A generated cluster unique Id created from cluster mentions ids
        """
        return '$'.join([mention.mention_id for mention in self.mentions])


class Clusters(object):
    cluster_coref_chain = 1000

    def __init__(self, topic_id: str, mentions: List[MentionData] = None) -> None:
        """

        Args:
            mentions: ``list[MentionData]``, required
                The initial mentions to create the clusters from
        """
        self.clusters_list = []
        self.topic_id = topic_id
        self.set_initial_clusters(mentions)

    def set_initial_clusters(self, mentions: List[MentionData]) -> None:
        """

        Args:
            mentions: ``list[MentionData]``, required
                The initial mentions to create the clusters from

        """
        if mentions:
            for mention in mentions:
                cluster = Cluster(Clusters.cluster_coref_chain)
                cluster.add_mention(mention)
                self.clusters_list.append(cluster)
                Clusters.cluster_coref_chain += 1

    def clean_clusters(self) -> None:
        """
        Remove all clusters that were already merged with other clusters
        """

        self.clusters_list = [cluster for cluster in self.clusters_list if not cluster.merged]

    def set_coref_chain_to_mentions(self) -> None:
        """
        Give all cluster mentions the same coref ID as cluster coref chain ID

        """
        for cluster in self.clusters_list:
            for mention in cluster.mentions:
                mention.predicted_coref_chain = str(cluster.coref_chain)

    def add_cluster(self, cluster: Cluster) -> None:
        self.clusters_list.append(cluster)

    def add_clusters(self, clusters) -> None:
        for cluster in clusters.clusters_list:
            self.clusters_list.append(cluster)
