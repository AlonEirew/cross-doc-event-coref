import logging

from src.dataobjs.mention_data import MentionDataLight
from src.dt_system.computed_relation_extraction import ComputedRelationExtraction
from src.dt_system.relation_type_enum import RelationTypeEnum


def run_example():
    logger.info('Running relation extraction example......')
    computed = ComputedRelationExtraction()

    mention_x1 = MentionDataLight(
        'IBM',
        mention_context='IBM manufactures and markets computer hardware, middleware and software'.split())
    mention_y1 = MentionDataLight(
        'International Business Machines',
        mention_context='International Business Machines Corporation is an '
                        'American multinational information technology company'.split())

    computed_relations = computed.extract_all_relations(mention_x1, mention_y1)

    if RelationTypeEnum.NO_RELATION_FOUND in computed_relations:
        logger.info('No Computed relation found')
    else:
        logger.info('Found Computed relations-%s', str(list(computed_relations)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    run_example()
