import logging
from typing import List

from src.ext_resources.relations.relation_extraction import RelationExtraction
from src.obj.sieves_config import EventSievesConfiguration, EntitySievesConfiguration

logger = logging.getLogger(__name__)


class SievesContainerInitialization(object):
    def __init__(self, event_coref_config: EventSievesConfiguration,
                 entity_coref_config: EntitySievesConfiguration,
                 sieves_model_list: List[RelationExtraction]):
        self.sieves_model_list = sieves_model_list
        self.event_config = event_coref_config
        self.entity_config = entity_coref_config

    def get_module_from_relation(self, relation_type):
        for model in self.sieves_model_list:
            if relation_type in model.get_supported_relations():
                return model

        raise Exception('No dl_model found that Support RelationType-' + str(relation_type))
