from .cityscapes import CitySegmentation
from .water import WaterSegmentation
from .water_val import CustomSegmentationDataset

datasets = {
    'citys': CitySegmentation,
    'water': WaterSegmentation,
    'water_val': CustomSegmentationDataset,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
