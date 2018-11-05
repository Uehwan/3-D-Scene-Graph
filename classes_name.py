import sys
sys.path.append('./FactorizableNet')
import random
import numpy.random as npr
import numpy as np
import argparse
import yaml
from pprint import pprint
import cv2
import torch
from torch.autograd import Variable
from lib import network
import lib.datasets as datasets
import lib.utils.general_utils as utils
import models as models
from models.HDN_v2.utils import interpret_relationships
import warnings
from settings import parse_args, testImageLoader
from PIL import Image
from sort.sort import Sort,iou
import os.path as osp
import interpret
import vis_V4
from keyframe_extracion import keyframe_checker
from SGGenModel import SGGen_MSDN, SGGen_DR_NET
args = parse_args()
# Set the random seed
random.seed(args.seed)
torch.manual_seed(args.seed + 1)
torch.cuda.manual_seed(args.seed + 2)
colorlist = [(random.randint(0,230),random.randint(0,230),random.randint(0,230)) for i in range(10000)]

# Set options
options = {
    'data': {
        'dataset_option': args.dataset_option,
        'batch_size': args.batch_size,
    },
}
with open(args.path_opt, 'r') as handle:
    options_yaml = yaml.load(handle)
options = utils.update_values(options, options_yaml)
with open(options['data']['opts'], 'r') as f:
    data_opts = yaml.load(f)
    options['data']['dataset_version'] = data_opts.get('dataset_version', None)
    options['opts'] = data_opts

print '## args'
pprint(vars(args))
print '## options'
pprint(options)
# To set the random seed
random.seed(args.seed)
torch.manual_seed(args.seed + 1)
torch.cuda.manual_seed(args.seed + 2)

print("Loading training set and testing set..."),
test_set = getattr(datasets, options['data']['dataset'])(data_opts, 'test',
                                                         dataset_option=options['data'].get('dataset_option', None),
                                                         batch_size=options['data']['batch_size'],
                                                         use_region=options['data'].get('use_region', False))
print("Done")



