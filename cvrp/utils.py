import pydoc

from typing import Any, Dict, List
from omegaconf import OmegaConf
from loguru import logger


def load_class_from_config(conf, **kwargs):
    logger.info(str(conf.cls))

    clazz = pydoc.locate(conf.cls)
    if clazz is None:
        raise ValueError(f"Cannot find class {conf.cls}")
    return clazz(**conf.kwargs, **kwargs)


def load_config(paths: List[str]):
    assert len(paths) > 0
    conf = OmegaConf.load(paths[0])

    for path in paths[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(path))

    return conf


def create_summary_index(instances: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}

    for inst in instances:
        index[inst["instance_name"]] = inst

    return index


def group_summaries_by_name(
    instance_list: List[List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}

    for instances in instance_list:
        for inst in instances:
            instance_name = inst["instance_name"]
            if instance_name not in groups.keys():
                groups[instance_name] = []
            groups[instance_name].append(inst)

    return groups
