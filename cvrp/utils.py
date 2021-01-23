import pydoc

from typing import List
from omegaconf import OmegaConf
from loguru import logger


def load_class_from_config(conf):
    logger.info(str(conf.cls))

    clazz = pydoc.locate(conf.cls)
    if clazz is None:
        raise ValueError(f"Cannot find class {conf.cls}")
    return clazz(**conf.kwargs)


def load_config(paths: List[str]):
    assert len(paths) > 0
    conf = OmegaConf.load(paths[0])

    for path in paths[1:]:
        conf = OmegaConf.merge(conf, OmegaConf.load(path))

    return conf
