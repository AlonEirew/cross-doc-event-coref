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

    def add_mention(self, mention) -> None:
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
    cluster_coref_chain = 0

    def __init__(self, topic_id: str, mentions=None) -> None:
        """

        Args:
            mentions: ``list[MentionData]``, required
                The initial mentions to create the clusters from
        """
        self.clusters_list = []
        self.topic_id = topic_id
        self.set_initial_clusters(mentions)

    def set_initial_clusters(self, mentions) -> None:
        """

        Args:
            mentions: ``list[MentionData]``, required
                The initial mentions to create the clusters from

        """
        if mentions is not None:
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

    def get_mentions(self):
        all_mentions = list()
        for cluster in self.clusters_list:
            all_mentions.extend(cluster.mentions)

        return all_mentions

    @staticmethod
    def from_clusters_to_mentions_list(clusters_list):
        """
        Args:
            clusters_list : List[Clusters]
        Returns:
            List[MentionData]
        """
        all_mentions = list()
        for clusters in clusters_list:
            for cluster in clusters.clusters_list:
                all_mentions.extend(cluster.mentions)

        all_mentions.sort(key=lambda mention: mention.mention_index)

        print(str(len(all_mentions)))

        return all_mentions

    @staticmethod
    def print_cluster_results(clusters, eval_type: str):
        """
        :param clusters: List[Clusters]
        :param eval_type: type of evaluation (eg. Event/Entity)
        :return:
        """
        print('-=' + eval_type + ' Clusters=-')
        for topic_cluster in clusters:
            print('\n\tTopic=' + topic_cluster.topic_id)
            for cluster in topic_cluster.clusters_list:
                cluster_mentions = list()
                for mention in cluster.mentions:
                    mentions_dict = dict()
                    mentions_dict['id'] = mention.mention_id
                    mentions_dict['text'] = mention.tokens_str
                    cluster_mentions.append(mentions_dict)

                print('\t\tCluster(' + str(cluster.coref_chain) + ') Mentions='
                      + str(cluster_mentions))

    @staticmethod
    def inc_cluster_coref_chain(value):
        Clusters.cluster_coref_chain += value

    @staticmethod
    def get_cluster_coref_chain():
        return Clusters.cluster_coref_chain

    @staticmethod
    def from_mentions_to_predicted_clusters(mentions):
        clusters = dict()
        for mention in mentions:
            if mention.predicted_coref_chain not in clusters:
                clusters[mention.predicted_coref_chain] = list()
            clusters[mention.predicted_coref_chain].append(mention)

        return clusters
