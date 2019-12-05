import logging
from typing import List

from src.dataobjs.sieves_config import EventSievesConfiguration, EntitySievesConfiguration
from src.dt_system.relation_extraction import RelationExtraction

logger = logging.getLogger(__name__)


class SievesContainerInitialization(object):
    def __init__(self, event_coref_config: EventSievesConfiguration = None,
                 entity_coref_config: EntitySievesConfiguration = None,
                 sieves_model_list: List[RelationExtraction] = None):
        self.sieves_model_list = sieves_model_list
        self.event_config = event_coref_config
        self.entity_config = entity_coref_config

    def get_module_from_relation(self, relation_type):
        for model in self.sieves_model_list:
            if relation_type in model.get_supported_relations():
                return model

        raise Exception('No dl_model found that Support RelationType-' + str(relation_type))
