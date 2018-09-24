from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
import os
import os.path as osp
import numpy as np

class AttrDict(dict):

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value
__C = AttrDict()
cfg = __C

__C.output_dir = 'output'
__C.name  = ''
__C.date = ''
__C.model = 'RFB_Net_vgg'
__C.cuda = True
__C.ngpu = 1
__C.num_workers = 8
__C.image_size = [300,300]
__C.phase = []
# network related params
__C.network = AttrDict()
__C.network.basenet = ''
__C.network.pretrained_epoch = 0
__C.network.rgb_means = [0, 0, 0]
__C.network.rgb_std = [1, 1, 1]
__C.network.multi_box_loss_type = 'origin'
# anchor related param
__C.anchor = AttrDict()
__C.anchor.feature_maps = [38, 19, 10, 5, 3, 1]
__C.anchor.min_dim = 300
__C.anchor.steps = [8, 16, 32, 64, 100, 300]
__C.anchor.min_sizes = [30, 60, 111, 162, 213, 264]
__C.anchor.max_sizes = [60, 111, 162, 213, 264, 315]
__C.anchor.aspect_ratios = [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
__C.anchor.variance = [0.1, 0.2]
__C.anchor.clip = True

# training related params
__C.train = AttrDict()
__C.train.lr_scheduler = AttrDict()
__C.train.lr_scheduler.lr = 0.001
__C.train.lr_scheduler.lr_decay_type = 'multi-step'
# multi step
__C.train.lr_scheduler.steps = []
__C.train.lr_scheduler.gamma = 0.1
# rmsprop
__C.train.lr_scheduler.alpha = 0.99
# adam
__C.train.lr_scheduler.beta1 = 0.99
__C.train.lr_scheduler.beta2 = 0.99
#warm up
__C.train.lr_scheduler.warmup = False
__C.train.lr_scheduler.warmup_step = 0
__C.train.lr_scheduler.begin_epoch = 0
__C.train.lr_scheduler.max_epochs = 0

__C.train.batch_size = 32
__C.train.save_frequency = 10
__C.train.log_iters = True
__C.train.checkpoint = ''
__C.train.resume_epoch = 0

__C.train.optimizer = AttrDict()
__C.train.optimizer.optimizer = 'sgd'
__C.train.optimizer.lr =  __C.train.lr_scheduler.lr
__C.train.optimizer.weight_decay = 0.0005
__C.train.optimizer.momentum = 0.9
__C.train.optimizer.eps = 1e-8
__C.train.train_scope = ''
__C.train.resume_scope = ''

# training related params
__C.test = AttrDict()
__C.test.NMS = 0.3
__C.test.max_per_image = 300
__C.test.test_frequency = 10
__C.test.retest = False
__C.test.batch_size = 1

# dataset related params
__C.dataset = AttrDict()
__C.dataset.dataset = 'VOC'
__C.dataset.dataset_dir = './data'
__C.dataset.num_classes = 21
__C.dataset.train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# test set scope
__C.dataset.test_sets = [('2007', 'test')]
# image expand probability during train
__C.dataset.zoom_p = 0.6
# image size
__C.dataset.image_size = __C.image_size
# train batch size
__C.dataset.train_batch_size = __C.train.batch_size
# test batch size
__C.dataset.test_batch_size = __C.test.batch_size
# number of workers to extract datas
__C.dataset.num_workers = 8
__C.dataset.image_size = __C.image_size
__C.dataset.rgb_means = __C.network.rgb_means
__C.dataset.rgb_std = __C.network.rgb_std


# matcher related params
__C.matcher = AttrDict()
__C.matcher.num_classes = __C.dataset.num_classes
__C.matcher.background_label = 0
__C.matcher.pos_th = 0.5
__C.matcher.neg_th= 0.5
__C.matcher.negpos_ratio = 3
__C.matcher.variance = [0.1, 0.2]

# post process related params
__C.post_process = AttrDict()
__C.post_process.num_classes = __C.dataset.num_classes
__C.post_process.background_label = __C.matcher.background_label
__C.post_process.score_threshold = 0.01
__C.post_process.nms = 0.45
__C.post_process.max_per_image = 100
__C.post_process.variance = __C.matcher.variance


def _merge_a_into_b(a, b, stack=None):
    """Merge __C dictionary a into __C dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
          raise KeyError('Non-existent config key: {}'.format(full_key))

        v = _decode_cfg_value(v_)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def update_cfg():
    __C.dataset.image_size = __C.image_size
    __C.dataset.train_batch_size = __C.train.batch_size
    __C.dataset.test_batch_size = __C.test.batch_size
    __C.matcher.num_classes = __C.dataset.num_classes
    __C.post_process.num_classes = __C.dataset.num_classes
    __C.post_process.background_label = __C.matcher.background_label
    __C.post_process.variance = __C.matcher.variance
    __C.train.optimizer.lr =  __C.train.lr_scheduler.lr
    __C.dataset.rgb_means = __C.network.rgb_means
    __C.dataset.rgb_std = __C.network.rgb_std


def cfg_from_file(filename):
    """Load a __C file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
    update_cfg()

def _decode_cfg_value(v):
    """Decodes a raw __C value (e.g., from a yaml __C files or command
    line argument) into a Python object.
    """
    # __Cs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v

def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config'
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
