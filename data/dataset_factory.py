from data import voc0712
from data import coco
dataset_map = {
                'voc0712': voc0712.VOCDetection,
                'coco': coco.COCODetection,
            }

def gen_dataset_fn(name):
    """Returns a dataset func.

    Args:
    name: The name of the dataset.

    Returns:
    func: dataset_fn

    Raises:
    ValueError: If network `name` is not recognized.
    """
    if name not in dataset_map:
        raise ValueError('The dataset unknown %s' % name)
    func = dataset_map[name]
    return func


import torch
import numpy as np

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

from data.data_augment import preproc
import torch.utils.data as data

def load_data(cfg, phase):
    if phase == 'train':
        dataset = dataset_map[cfg.dataset](cfg.dataset_dir, cfg.train_sets, preproc(cfg.image_size[0], cfg.rgb_means, cfg.rgb_std, cfg.zoom_p))
        data_loader = data.DataLoader(dataset, cfg.train_batch_size, num_workers=cfg.num_workers,
                                  shuffle=True, collate_fn=detection_collate, pin_memory=True)
    if phase == 'eval':
        dataset = dataset_map[cfg.dataset](cfg.dataset_dir, cfg.test_sets, None)
        data_loader = data.DataLoader(dataset, cfg.test_batch_size, num_workers=cfg.num_workers,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    if phase == 'test':
        dataset = dataset_map[cfg.dataset](cfg.dataset_dir, cfg.test_sets, None)
        data_loader = data.DataLoader(dataset, cfg.test_batch_size, num_workers=cfg.num_workers,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    if phase == 'visualize':
        dataset = dataset_map[cfg.dataset](cfg.dataset_dir, cfg.test_sets, None)
        data_loader = data.DataLoader(dataset, cfg.test_batch_size, num_workers=cfg.num_workers,
                                  shuffle=False, collate_fn=detection_collate, pin_memory=True)
    return data_loader
