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
import interpret
from vis import vis_object_detection, scene_graph
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

# Model declaration
model = getattr(models, options['model']['arch'])(test_set, opts=options['model'])
print("Done.")
test_loader = torch.utils.data.DataLoader(test_set, batch_size=options['data']['batch_size'],
                                          shuffle=False, num_workers=args.workers,
                                          pin_memory=True,
                                          collate_fn=getattr(datasets, options['data']['dataset']).collate)

network.set_trainable(model, False)
print('Loading pretrained model: {}'.format(args.pretrained_model))
args.train_all = True
network.load_net(args.pretrained_model, model)
# Setting the state of the training model
model.cuda()
model.eval()

print('--------------------------------------------------------------------------')
print('3D-Scene-Graph-Generator Demo: Object detection and Scene Graph Generation')
print('--------------------------------------------------------------------------')
imgLoader = testImageLoader(args)
# Initial Sort tracker
interpreter = interpret.interpreter(args,ENABLE_TRACKING=False)

try:
    for idx in range(imgLoader.num_frames)[700:]:
    #for idx in range(1000000000)[0:]:
    #for idx in [100,118,119,162,163,195,212,2223,232,498]:
        print('...........................................................................')
        print('Image '+str(idx))
        print('...........................................................................')
        ''' 1. Load an color/depth image and camera parameter'''
        if args.dataset == 'scannet':
            image_scene, depth_img, pix_depth, inv_p_matrix, inv_R, Trans = imgLoader.load_image(frame_idx=idx)
        else:
            image_scene = imgLoader.load_image(frame_idx=idx)
            if type(image_scene) !=np.ndarray:
                continue
            depth_img, pix_depth, inv_p_matrix, inv_R, Trans = None, None, None, None, None
        img_original_shape = image_scene.shape
        #ratio = float(image_scene.shape[0]) / float(image_scene.shape[1])
        #if ratio>0.4:
        #    continue
        cv2.imwrite('./vis_result/raw'+str(idx)+'.jpg',image_scene)
        ''' 2. Pre-processing: Rescale & Normalization '''
        # Resize the image to target scale
        if args.dataset == 'scannet':
            image_scene= cv2.resize(image_scene, (depth_img.size), interpolation= cv2.INTER_AREA)
        target_scale = test_set.opts[test_set.cfg_key]['SCALES'][npr.randint(0, high=len(test_set.opts[test_set.cfg_key]['SCALES']))]
        im_data, im_scale = test_set._image_resize(image_scene, target_scale, test_set.opts[test_set.cfg_key]['MAX_SIZE'])
        # restore the [image_height, image_width, scale_factor, max_size]
        im_info = np.array([[im_data.shape[0], im_data.shape[1], im_scale,
                                       img_original_shape[0], img_original_shape[1]]], dtype=np.float)
        im_data = Image.fromarray(im_data)
        im_data = test_set.transform(im_data) # normalize the image with the pre-defined min/std.
        im_data = Variable(im_data.cuda(), volatile=True).unsqueeze(0)

        ''' 3. Object Detection & Scene Graph Generation from the Pre-trained MSDN Model '''
        object_result, predicate_result = model.forward_eval(im_data, im_info, )

        ''' 4. Post-processing: Interpret the Model Output '''
        # interpret the model output
        obj_boxes, obj_scores, obj_cls, \
        subject_inds, object_inds, \
        subject_boxes, object_boxes, \
        subject_IDs, object_IDs, \
        predicate_inds, triplet_scores, relationships = \
            interpreter.interpret_graph(object_result, predicate_result,im_info)



        image_scene = vis_object_detection(image_scene, test_set, obj_cls, obj_boxes, obj_scores)
        cv2.imshow('sample', image_scene)
        cv2.imwrite('./vis_result/detected'+str(idx)+'.jpg',image_scene)

        print('-------Subject-------|------Predicate-----|--------Object---------|--Score-')
        for i,relation in enumerate(relationships):
            if relation[1] > 0:  # '0' is the class 'irrelevant'
                print('{sbj_cls:9} {sbj_ID:4} {sbj_score:1.2f}  |  '
                      '{pred_cls:11} {pred_score:1.2f}  |  '
                      '{obj_cls:9} {obj_ID:4} {obj_score:1.2f}  |  '
                      '{triplet_score:1.3f}'.format(
                    sbj_cls=test_set.object_classes[int(obj_cls[int(relation[0])])], sbj_score=obj_scores[int(relation[0])],
                    sbj_ID=str(int(obj_boxes[int(relation[0])][4])),
                    #sbj_ID=str(subject_IDs[i]),

                    pred_cls=test_set.predicate_classes[int(relation[2])],
                    pred_score=relation[3] / obj_scores[int(relation[0])] / obj_scores[int(relation[1])],
                    obj_cls=test_set.object_classes[int(obj_cls[int(relation[1])])], obj_score=obj_scores[int(relation[1])],
                    obj_ID=str(int(obj_boxes[int(relation[1])][4])),
                    #obj_ID=str(object_IDs[i]),

                    triplet_score=relation[3]))

        with open('./vis_result/graph'+str(idx)+'.txt', 'w') as f:
            f.write('-------Subject-------|------Predicate-----|--------Object---------|--Score-\n')
            for i,relation in enumerate(relationships):
                if relation[1] > 0:  # '0' is the class 'irrelevant'
                    f.write('{sbj_cls:9} {sbj_ID:4} {sbj_score:1.2f}  |  '
                          '{pred_cls:11} {pred_score:1.2f}  |  '
                          '{obj_cls:9} {obj_ID:4} {obj_score:1.2f}  |  '
                          '{triplet_score:1.3f}\n'.format(
                        sbj_cls=test_set.object_classes[int(obj_cls[int(relation[0])])], sbj_score=obj_scores[int(relation[0])],
                        sbj_ID=str(int(obj_boxes[int(relation[0])][4])),
                        #sbj_ID=str(subject_IDs[i]),

                        pred_cls=test_set.predicate_classes[int(relation[2])],
                        pred_score=relation[3] / obj_scores[int(relation[0])] / obj_scores[int(relation[1])],
                        obj_cls=test_set.object_classes[int(obj_cls[int(relation[1])])], obj_score=obj_scores[int(relation[1])],
                        obj_ID=str(int(obj_boxes[int(relation[1])][4])),
                        #obj_ID=str(object_IDs[i]),
                        triplet_score=relation[3]))

        # print('-------Subject-------|------Predicate-----|--------Object---------|--Score-')
        # for i,relation in enumerate(relationships):
        #     if relation[2] > 0:  # '0' is the class 'irrelevant'
        #         print('{sbj_cls:9} {sbj_ID:4} {sbj_score:1.2f}  |  '
        #               '{pred_cls:11} {pred_score:1.2f}  |  '
        #               '{obj_cls:9} {obj_ID:4} {obj_score:1.2f}  |  '
        #               '{triplet_score:1.3f}'.format(
        #             sbj_cls=test_set.object_classes[subject_inds[i]], sbj_score=obj_scores[int(relation[0])],
        #             #sbj_ID=str(int(obj_boxes[int(relation[0])][4])),
        #             sbj_ID=str(subject_IDs[i]),
        #             pred_cls=test_set.predicate_classes[predicate_inds[i]],
        #             pred_score=relation[3] / obj_scores[int(relation[0])] / obj_scores[int(relation[1])],
        #             obj_cls=test_set.object_classes[object_inds[i]], obj_score=obj_scores[int(relation[1])],
        #             #obj_ID=str(int(obj_boxes[int(relation[1])][4])),
        #             obj_ID=str(object_IDs[i]),
        #             triplet_score=triplet_scores[i]))


        cv2.waitKey(1)
        #cv2.destroyAllWindows()
        #cv2.destroyAllWindows()
except KeyboardInterrupt:
    pass











