import os
import random
import numpy as np
import numpy.random as npr
import argparse
import torch
from faster_rcnn import network
from faster_rcnn.MSDN import Hierarchical_Descriptive_Model
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.config import cfg
from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.utils.HDN_utils import get_model_name, group_features
from PIL import Image
import os.path as osp
import cv2
import vis
from sort.sort import Sort, iou
from settings import parse_args, testImageLoader
import pandas as pd
from graphviz import Digraph
TIME_IT = cfg.TIME_IT

args = parse_args()
args.resume_training=True
args.resume_model = './pretrained_models/HDN_1_iters_alt_normal_I_LSTM_with_bias_with_dropout_0_5_nembed_256_nhidden_512_with_region_regression_resume_SGD_best.h5'
args.dataset_option = 'normal'
args.MPS_iter =1
args.caption_use_bias = True
args.caption_use_dropout = True
args.rnn_type = 'LSTM_normal'
args.evaluate = True

# To set the model name automatically
print(args)
lr = args.lr
args = get_model_name(args)
print('Model name: {}'.format(args.model_name))

# To set the random seed
random.seed(args.seed)
torch.manual_seed(args.seed + 1)
torch.cuda.manual_seed(args.seed + 2)
colorlist = [(random.randint(0,230),random.randint(0,230),random.randint(0,230)) for i in range(10000)]

print("Loading test_set..."),
test_set = visual_genome('small', 'test')
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
print("Done.")

# Model declaration
net = Hierarchical_Descriptive_Model(nhidden=args.mps_feature_len,
             n_object_cats=test_set.num_object_classes,
             n_predicate_cats=test_set.num_predicate_classes,
             n_vocab=test_set.voc_size,
             voc_sign=test_set.voc_sign,
             max_word_length=test_set.max_size,
             MPS_iter=args.MPS_iter,
             use_language_loss=not args.disable_language_model,
             object_loss_weight=test_set.inverse_weight_object,
             predicate_loss_weight=test_set.inverse_weight_predicate,
             dropout=args.dropout,
             use_kmeans_anchors=not args.use_normal_anchors,
             gate_width = args.gate_width,
             nhidden_caption = args.nhidden_caption,
             nembedding = args.nembedding,
             rnn_type=args.rnn_type,
             rnn_droptout=args.caption_use_dropout, rnn_bias=args.caption_use_bias,
             use_region_reg = args.region_bbox_reg,
             use_kernel = args.use_kernel_function)
# params = list(net.parameters())
# for param in params:
#     print param.size()
# print net

# Set the state of the trained model
net.cuda()
net.eval()
network.set_trainable(net, False)
network.load_net(args.resume_model, net)
target_scale = cfg.TRAIN.SCALES[npr.randint(0, high=len(cfg.TRAIN.SCALES))]  # target_scale = 600. why?

print('-------------------------------------------------------------------------')
print('MSDN Demo: Object detection and Scene Graph Generation')
print('-------------------------------------------------------------------------')
imgLoader = testImageLoader(args)
# Initial Sort tracker
tracker = Sort()
scene_graph = vis.scene_graph()

for idx in range(imgLoader.num_frames)[0:]:
    print('...........................................................................')
    print('Image '+str(idx))
    print('...........................................................................')
    ''' 1. Load an color/depth image and camera parameter'''
    if args.dataset == 'scannet':
        image_scene, depth_img, pix_depth, inv_p_matrix, inv_R, Trans = imgLoader.gen_img_path(frame_idx=idx)
    else:
        image_scene = imgLoader.gen_img_path(frame_idx=idx)
        depth_img, pix_depth, inv_p_matrix, inv_R, Trans = None, None, None, None, None


    ''' 2. Rescale & Normalization '''
    # Resize the image to target scale
    if args.dataset == 'scannet':
        image_scene= cv2.resize(image_scene, (depth_img.size), interpolation= cv2.INTER_AREA)
        image_caption = image_scene.copy()
        #cv2.imshow('image_scene', image_scene)
        im_data, im_scale = test_set._image_resize(image_caption, target_scale, cfg.TRAIN.MAX_SIZE)
    else:
        image_caption = image_scene.copy()
        im_data, im_scale = test_set._image_resize(image_caption, target_scale, cfg.TRAIN.MAX_SIZE)

    im_info = torch.FloatTensor([[im_data.shape[0], im_data.shape[1], im_scale]]) # image shape, scale ratio(resize)
    if test_set.transform is not None:
        im_data = test_set.transform(im_data) # normalize the image with the pre-defined min/std.

    # _annotation = test_set.annotations[idx]
    # gt_boxes_object = torch.zeros((len(_annotation['objects']), 5))
    # gt_boxes_region = torch.zeros((len(_annotation['regions']), test_set.max_size + 4)) # 4 for box and 40 for sentences
    # gt_boxes_object[:, 0:4] = torch.FloatTensor([obj['box'] for obj in _annotation['objects']]) * im_scale
    # gt_boxes_region[:, 0:4] = torch.FloatTensor([reg['box'] for reg in _annotation['regions']]) * im_scale
    # gt_boxes_object[:, 4]   = torch.FloatTensor([obj['class'] for obj in _annotation['objects']])
    # gt_boxes_region[:, 4:]  = torch.FloatTensor([np.pad(reg['phrase'],
    #                             (0,test_set.max_size-len(reg['phrase'])),'constant',constant_values=test_set.voc_sign['end'])
    #                                 for reg in _annotation['regions']])
    # gt_relationships = torch.zeros(len(_annotation['objects']), (len(_annotation['objects']))).type(torch.LongTensor)
    # for rel in _annotation['relationships']:
    #     gt_relationships[rel['sub_id'], rel['obj_id']] = rel['predicate']


    ''' 3. Object Detection & Scene Graph Generation from the Pre-trained MSDN Model '''
    object_result, predicate_result, region_result = net.forward(im_data.unsqueeze(0),im_info,graph_generation=True)
    cls_prob_object, bbox_object, object_rois = object_result[:]
    cls_prob_predicate, mat_phrase = predicate_result[:]
    region_caption, bbox_region, region_rois, region_scores = region_result[:]

    ''' 4. Post-processing: Interpret the Model Output '''
    # interpret the model output
    obj_boxes, obj_scores, obj_inds, \
    subject_inds,  object_inds, \
    subject_boxes, object_boxes, \
    subject_IDs,   object_IDs, \
    predicate_inds, triplet_scores = \
        net.interpret_graph(cls_prob_object, bbox_object, object_rois,
                            cls_prob_predicate, mat_phrase, im_info,
                            nms=True, top_N=args.top_N_triplets, use_gt_boxes=False)
    region_caption, region_scores, region_boxes = \
        net.interpret_caption(region_caption, bbox_region, region_rois,
                              region_scores, im_info, top_N=args.top_N_captions)

    ''' 5. Print Scene Graph '''
    image_scene = vis.vis_object_detection(image_scene, test_set,
                    obj_inds, obj_boxes, obj_scores)

    scene_graph.vis_scene_graph(idx, test_set,
                    obj_inds, obj_boxes, obj_scores,
                    subject_inds, predicate_inds, object_inds,
                    subject_IDs, object_IDs, triplet_scores,
                    pix_depth, inv_p_matrix, inv_R, Trans, dataset=args.dataset)
    winname_scene = 'image_scene'
    cv2.namedWindow(winname_scene)  # Create a named window
    cv2.moveWindow(winname_scene, 10, 10)
    cv2.imshow(winname_scene, image_scene)
    cv2.imwrite(osp.join(args.vis_result_path, str(idx) + '.jpg'), image_scene)

    ''' 6. Print Captions '''
    image_caption = vis.vis_caption(image_caption, test_set, region_caption, region_boxes, region_scores)
    winname_caption = 'image_caption'
    cv2.namedWindow(winname_caption)  # Create a named window
    cv2.moveWindow(winname_caption, 10, 500)
    cv2.imshow(winname_caption, image_caption)
    cv2.waitKey(0)
    cv2.destroyAllWindows()














