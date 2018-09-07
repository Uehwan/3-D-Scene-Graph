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
tracker = Sort()
#scene_graph = vis.scene_graph()

def filter_untracted(ref_bbox, tobefiltered_bbox):
    keep = []
    for bbox in ref_bbox:
        ious = [iou(bbox[:4], obj_box) for obj_box in tobefiltered_bbox]
        keep.append(np.argmax(ious))
    return np.array(keep)
def get_class_string(class_index, score, dataset):
    class_text = dataset[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_object_detection_(image_scene, test_set,
                         obj_inds, obj_boxes, obj_scores):
    for i, obj_ind in enumerate(obj_inds):
        cv2.rectangle(image_scene,
                      (int(obj_boxes[i][0]), int(obj_boxes[i][1])),
                      (int(obj_boxes[i][2]), int(obj_boxes[i][3])),
                      colorlist[i],
                      2)
        font_scale = 0.5
        txt = str(i) + '. ' + get_class_string(obj_ind, obj_scores[i], test_set.object_classes)
        ((txt_w, txt_h), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # Place text background.
        x0, y0 = int(obj_boxes[i][0]), int(obj_boxes[i][3])
        back_tl = x0, y0 - int(1.3 * txt_h)
        back_br = x0 + txt_w, y0
        cv2.rectangle(image_scene, back_tl, back_br, colorlist[i], -1)
        cv2.putText(image_scene,
                    txt,
                    (x0, y0 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1)
    return image_scene

def vis_object_detection(image_scene,test_set,
                    obj_inds,obj_boxes,obj_scores):

    for i, obj_ind in enumerate(obj_inds):
        cv2.rectangle(image_scene,
                      (int(obj_boxes[i][0]), int(obj_boxes[i][1])),
                      (int(obj_boxes[i][2]), int(obj_boxes[i][3])),
                      colorlist[int(obj_boxes[i][4])],
                      2)
        font_scale=0.5
        txt =str(int(obj_boxes[i][4])) + '. '+get_class_string(obj_ind,obj_scores[i],test_set.object_classes)
        ((txt_w, txt_h), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # Place text background.
        x0, y0 = int(obj_boxes[i][0]),int(obj_boxes[i][3])
        back_tl = x0, y0 - int(1.3 * txt_h)
        back_br = x0 + txt_w, y0
        cv2.rectangle(image_scene, back_tl, back_br, colorlist[int(obj_boxes[i][4])], -1)
        cv2.putText(image_scene,
                    txt,
                    (x0,y0-2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255,255,255),
                    1)

    return image_scene

for idx in range(imgLoader.num_frames)[0:]:
    print('...........................................................................')
    print('Image '+str(idx))
    print('...........................................................................')
    ''' 1. Load an color/depth image and camera parameter'''
    if args.dataset == 'scannet':
        image_scene, depth_img, pix_depth, inv_p_matrix, inv_R, Trans = imgLoader.load_image(frame_idx=idx)
    else:
        image_scene = imgLoader.load_image(frame_idx=idx)
        depth_img, pix_depth, inv_p_matrix, inv_R, Trans = None, None, None, None, None

    img_original_shape = image_scene.shape
    ''' 2. Rescale & Normalization '''
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
    cls_prob_object, bbox_object, object_rois, reranked_score = object_result[:4]
    cls_prob_predicate, mat_phrase = predicate_result[:2]
    region_rois_num = predicate_result[2]

    ''' 4. Post-processing: Interpret the Model Output '''
    # interpret the model output
    obj_boxes, obj_scores, obj_cls, subject_inds, object_inds, \
    subject_boxes, object_boxes, predicate_inds, \
    sub_assignment, obj_assignment, total_score = \
        interpret_relationships(cls_prob_object, bbox_object, object_rois,
                                cls_prob_predicate, mat_phrase, im_info,
                                nms=args.nms, top_N=100,
                                use_gt_boxes=False,
                                triplet_nms=args.triplet_nms,
                                reranked_score=reranked_score)

    # filter out who has low obj_score
    keep_obj = np.where(obj_scores >= args.obj_thres)[0]
    if keep_obj.size == 0:
        print("no object detected ... continue to the next image")
        continue
    cutline_idx = max(keep_obj)
    obj_scores = obj_scores[:cutline_idx + 1]
    obj_boxes = obj_boxes[:cutline_idx + 1]
    obj_cls = obj_cls[:cutline_idx + 1]

    relationships = np.array(zip(sub_assignment, obj_assignment, predicate_inds, total_score))
    keep_sub_assign = np.where(relationships[:, 0] <= cutline_idx)[0]
    relationships = relationships[keep_sub_assign]
    keep_obj_assign = np.where(relationships[:, 1] <= cutline_idx)[0]
    relationships = relationships[keep_obj_assign]

    # filter out who has low total_score
    keep_rel = np.where(relationships[:, 3] >= args.triplet_thres)[0] # MSDN:0.02, DR-NET:0.03
    if keep_rel.size == 0:
        print("no relation detected ... continue to the next image")
        continue
    cutline_idx = max(keep_rel)
    relationships = relationships[:cutline_idx + 1]

    # Object tracking
    # Filter out all un-tracked objects and triplets
    tracking_input = np.concatenate((obj_boxes, obj_scores.reshape(len(obj_scores), 1)), axis=1)
    bboxes_and_uniqueIDs = tracker.update(tracking_input)
    keep = filter_untracted(bboxes_and_uniqueIDs, obj_boxes)

    keep_sub_assign = [np.where(relationships[:,0] == keep_idx) for keep_idx in keep]
    keep_sub_assign = np.concatenate(keep_sub_assign, axis=1).flatten()
    relationships = relationships[keep_sub_assign]

    keep_obj_assign = [np.where(relationships[:,1] == keep_idx) for keep_idx in keep]
    keep_obj_assign = np.concatenate(keep_obj_assign, axis=1).flatten()
    relationships = relationships[keep_obj_assign]

    for i,k in enumerate(keep):
        relationships[relationships==k]=i


    #predicate_inds = predicate_inds.squeeze()[pred_list]
    #subject_inds = obj_cls[relationships[:,0]]
    #object_inds = obj_cls[relationships[:,1]]
    obj_boxes = np.concatenate([obj_boxes, np.zeros([obj_boxes.shape[0], 1])], axis=1)
    for i, keep_idx in enumerate(keep):
        obj_boxes[keep_idx] = bboxes_and_uniqueIDs[i]
    subject_boxes = obj_boxes[relationships[:,0].astype(int)]
    object_boxes = obj_boxes[relationships[:,1].astype(int)]
    obj_scores = obj_scores[keep]
    obj_cls = obj_cls[keep]
    obj_boxes = bboxes_and_uniqueIDs

    subject_IDs = subject_boxes[:, 4].astype(int)
    object_IDs = object_boxes[:, 4].astype(int)
    print('-------Subject-------|------Predicate-----|--------Object---------|--Score-')
    for relation in relationships:
        if relation[1] > 0:  # '0' is the class 'irrelevant'
            print('{sbj_cls:9} {sbj_ID:4} {sbj_score:1.2f}  |  '
                  '{pred_cls:11} {pred_score:1.2f}  |  '
                  '{obj_cls:9} {obj_ID:4} {obj_score:1.2f}  |  '
                  '{triplet_score:1.3f}'.format(
                sbj_cls=test_set.object_classes[int(obj_cls[int(relation[0])])], sbj_score=obj_scores[int(relation[0])],
                sbj_ID=str(int(obj_boxes[int(relation[0])][4])),
                pred_cls=test_set.predicate_classes[int(relation[2])],
                pred_score=relation[3] / obj_scores[int(relation[0])] / obj_scores[int(relation[1])],
                obj_cls=test_set.object_classes[int(obj_cls[int(relation[1])])], obj_score=obj_scores[int(relation[1])],
                obj_ID=str(int(obj_boxes[int(relation[1])][4])),
                triplet_score=relation[3]))


    # #colorlist = [(random.randint(0, 210), random.randint(0, 210), random.randint(0, 210)) for i in range(10000)]
    image_scene = vis_object_detection(image_scene, test_set, obj_cls, obj_boxes, obj_scores)
    cv2.imshow('sample', image_scene)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()










