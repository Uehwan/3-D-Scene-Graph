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
import os.path as osp
import vis
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

# Model declaration
#model = getattr(models, options['model']['arch'])(test_set, opts=options['model'])
if args.path_opt.split('/')[-1].strip() == 'VG-DR-Net.yaml':
    model = SGGen_DR_NET(args, test_set, opts=options['model'])
elif args.path_opt.split('/')[-1].strip() == 'VG-MSDN.yaml':
    model = SGGen_MSDN(args, test_set, opts=options['model'])
else:
    raise NotImplementedError
print("Done.")
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
scene_graph = vis.scene_graph()
keyframe_extractor = keyframe_checker(args,
                                      intrinsic_depth=imgLoader.intrinsic_depth,
                                      alpha=0.4,
                                      blurry_gain=30,
                                      blurry_offset=10)

for idx in range(imgLoader.num_frames)[0:]:
    #for idx in range(1000000000)[0:]:
    #for idx in [100,118,119,162,163,195,212,2223,232,498]:
    print('...........................................................................')
    print('Image '+str(idx))
    print('...........................................................................')
    ''' 1. Load an color/depth image and camera parameter'''
    if args.dataset == 'scannet':
        image_scene, depth_img, pix_depth, inv_p_matrix, inv_R, Trans, camera_pose = imgLoader.load_image(frame_idx=idx)
    else:
        image_scene = imgLoader.load_image(frame_idx=idx)
        if type(image_scene) !=np.ndarray:
            continue
        depth_img, pix_depth, inv_p_matrix, inv_R, Trans, camera_pose = None, None, None, None, None, None
    img_original_shape = image_scene.shape
    #ratio = float(image_scene.shape[0]) / float(image_scene.shape[1])
    #if ratio>0.4:
    #    continue
    #cv2.imwrite('./vis_result/raw'+str(idx)+'.jpg',image_scene)

    ''' 3. Pre-processing: Rescale & Normalization '''
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

    ''' 2. Key-frame Extraction: Check if this frame is key-frame or anchor-frame'''
    #if args.dataset == "scannet":
    IS_KEY_OR_ANCHOR, blurry_score, blurry_thres = keyframe_extractor.check_frame(image_scene, depth_img, camera_pose)
    winname_scene = '{idx:4}. blur_score: {score:4.2f}, blur_thres: {thres:4.1f}, KEY_OR_ANCHOR: {flag:5}' \
        .format(idx=idx, score=blurry_score, thres=blurry_thres, flag=str(IS_KEY_OR_ANCHOR))
    print(winname_scene)
    if not IS_KEY_OR_ANCHOR:
        cv2.namedWindow('sample')  # Create a named window
        cv2.moveWindow('sample', 10, 10)
        cv2.putText(image_scene,
                    winname_scene,
                    (1,11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    #(150, 150, 150),
                    (30, 30, 200),
                    2)
        cv2.imshow('sample', image_scene)
        cv2.waitKey(1)
        continue

    ''' 4. Object Detection & Scene Graph Generation from the Pre-trained MSDN Model '''
    object_result, predicate_result = model.forward_eval(im_data, im_info, )


    ''' 5. Post-processing: Interpret the Model Output '''
    # interpret the model output
    obj_boxes, obj_scores, obj_cls, \
    subject_inds, object_inds, \
    subject_boxes, object_boxes, \
    subject_IDs, object_IDs, \
    predicate_inds, triplet_scores, relationships = \
        interpreter.interpret_graph(object_result, predicate_result,im_info)

    # sbj_scores = [obj_scores[int(relation[0])] for relation in relationships]
    # pred_scores = [relation[3] / obj_scores[int(relation[0])] / obj_scores[int(relation[1])] for relation in relationships]
    # obj_scores = [obj_scores[int(relation[1])] for relation in relationships]
    # triplet_scores = zip(sbj_scores, pred_scores, obj_scores)

    ''' 6. Print Scene Graph '''
    # original image_scene
    image_scene = vis.vis_object_detection(image_scene, test_set, obj_cls[:,0], obj_boxes, obj_scores[:,0])
    cv2.namedWindow('sample')  # Create a named window
    cv2.moveWindow('sample', 10, 10)
    cv2.putText(image_scene,
                winname_scene,
                (1, 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                #(150, 150, 150),
                (30, 200, 30),
                2)
    cv2.imshow('sample', image_scene)
    cv2.imwrite(osp.join(args.vis_result_path, str(idx) + '.jpg'), image_scene)

    print('-------Subject-------|------Predicate-----|--------Object---------|--Score-')
    for i, relation in enumerate(relationships):
        # print(test_set.object_classes[int(obj_cls[:, 0][int(relation[0])])])
        # print(obj_scores[:, 0][int(relation[0])])
        # print(str(int(obj_boxes[int(relation[0])][4])))
        #
        # print(str(int(obj_boxes[int(relation[1])][4])))
        # print(relation[3])
        if relation[1] > 0:  # '0' is the class 'irrelevant'
            print('{sbj_cls:9} {sbj_ID:4} {sbj_score:1.2f}  |  '
                  '{pred_cls:11} {pred_score:1.2f}  |  '
                  '{obj_cls:9} {obj_ID:4} {obj_score:1.2f}  |  '
                  '{triplet_score:1.3f}'.format(
                sbj_cls=test_set.object_classes[int(obj_cls[:,0][int(relation[0])])],
                sbj_score=obj_scores[:,0][int(relation[0])],
                sbj_ID=str(int(obj_boxes[int(relation[0])][4])),
                # sbj_ID=str(subject_IDs[i]),

                pred_cls=test_set.predicate_classes[int(relation[2])],
                pred_score=relation[3] / obj_scores[:,0][int(relation[0])] / obj_scores[:,0][int(relation[1])],
                obj_cls=test_set.object_classes[int(obj_cls[:,0][int(relation[1])])],
                obj_score=obj_scores[:,0][int(relation[1])],
                obj_ID=str(int(obj_boxes[int(relation[1])][4])),
                # obj_ID=str(object_IDs[i]),
                triplet_score=relation[3]))

    cv2.waitKey(2000)
    #cv2.destroyAllWindows()
    #cv2.destroyAllWindows()












