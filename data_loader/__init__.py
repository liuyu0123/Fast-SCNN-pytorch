from .cityscapes import CitySegmentation
from .water import WaterSegmentation

datasets = {
    'citys': CitySegmentation,
    'water': WaterSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
