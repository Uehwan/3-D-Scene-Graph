import sys
sys.path.append('./FactorizableNet')
import random
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

parser = argparse.ArgumentParser('Options for training Hierarchical Descriptive Model in pytorch')

parser.add_argument('--path_opt', default='options/models/VG-MSDN.yaml', type=str,
                    help='path to a yaml options file, VG-DR-Net.yaml or VG-MSDN.yaml')
parser.add_argument('--dataset_option', type=str, default='normal', help='data split selection [small | fat | normal]')
parser.add_argument('--batch_size', type=int, help='#images per batch')
parser.add_argument('--workers', type=int, default=4, help='#idataloader workers')
# model init
parser.add_argument('--pretrained_model', type=str, default = 'FactorizableNet/output/trained_models/Model-VG-MSDN.h5',
                    help='path to pretrained_model, Model-VG-DR-Net.h5 or Model-VG-MSDN.h5')

# structure settings
# Environment Settings
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--nms', type=float, default=0.3, help='NMS threshold for post object NMS (negative means not NMS)')
parser.add_argument('--triplet_nms', type=float, default=0.01, help='Triplet NMS threshold for post object NMS (negative means not NMS)')
# testing settings
parser.add_argument('--use_gt_boxes', action='store_true', help='Use ground truth bounding boxes for evaluation')
args = parser.parse_args()


def get_class_string(class_index, score, dataset):
    class_text = dataset[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def vis_object_detection(image_scene, test_set,
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


if __name__ == '__main__':

    # Set options
    options = {
        'data':{
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
    model = getattr(models, options['model']['arch'])(test_set, opts = options['model'])
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


    for i, sample in enumerate(test_loader): # (im_data, im_info, gt_objects, gt_relationships)
        if i < 500: continue
        im_data = Variable(sample['visual'].cuda(), volatile=True)
        gt_objects = sample['objects'][0]
        gt_relationships = sample['relations'][0]
        im_info = sample['image_info']


        object_result, predicate_result = model.forward_eval(im_data, im_info, )
        cls_prob_object, bbox_object, object_rois, reranked_score = object_result[:4]
        cls_prob_predicate, mat_phrase = predicate_result[:2]
        region_rois_num = predicate_result[2]
        # interpret the model output
        obj_boxes, obj_scores, obj_cls, subject_inds, object_inds, \
        subject_boxes, object_boxes, predicate_inds, \
        sub_assignment, obj_assignment, total_score = \
            interpret_relationships(cls_prob_object, bbox_object, object_rois,
                                    cls_prob_predicate, mat_phrase, im_info,
                                    nms=0.3, top_N=1,topk=1,
                                    use_gt_boxes=False,
                                    triplet_nms=0.01,
                                    reranked_score=reranked_score)

        # filter out who has low obj_score
        keep_obj=np.where(obj_scores>=0.2)[0]
        if keep_obj.size==0:
            warnings.warn("no object detected ... continue to the next image")
            continue
        cutline_idx = max(keep_obj)
        obj_scores = obj_scores[:cutline_idx+1]
        obj_boxes = obj_boxes[:cutline_idx+1]
        obj_cls = obj_cls[:cutline_idx+1]

        relationships = np.array(zip(sub_assignment, obj_assignment, predicate_inds, total_score))
        keep_sub_assign = np.where(relationships[:,0]<=cutline_idx)[0]
        relationships = relationships[keep_sub_assign]
        keep_obj_assign = np.where(relationships[:,1]<=cutline_idx)[0]
        relationships = relationships[keep_obj_assign]

        # filter out who has low total_score
        keep_rel = np.where(relationships[:,3]>=0.03)[0]
        if keep_rel.size == 0:
            warnings.warn("no relation detected ... continue to the next image")
            continue
        cutline_idx = max(keep_rel)
        relationships = relationships[:cutline_idx+1]




        print('-------Subject-------|------Predicate-----|--------Object---------|--Score-')
        for relation in relationships:
            if relation[2] > 0:  # '0' is the class 'irrelevant'
                print('{sbj_cls:9} {sbj_ID:4} {sbj_score:1.2f}  |  '
                      '{pred_cls:11} {pred_score:1.2f}  |  '
                      '{obj_cls:9} {obj_ID:4} {obj_score:1.2f}  |  '
                      '{triplet_score:1.3f}'.format(
                    sbj_cls=test_set.object_classes[int(obj_cls[int(relation[0])])], sbj_score=obj_scores[int(relation[0])],
                    sbj_ID=str(int(relation[0])),
                    pred_cls=test_set.predicate_classes[int(relation[2])], pred_score=relation[3]/obj_scores[int(relation[0])]/obj_scores[int(relation[1])],
                    obj_cls=test_set.object_classes[int(obj_cls[int(relation[1])])], obj_score=obj_scores[int(relation[1])],
                    obj_ID=str(int(relation[1])),
                    triplet_score=relation[3]))

            if relation[2]==9:
                sample_img_path = './data/svg/images/' + test_set.annotations[i]['path']
                img_scene = cv2.imread(sample_img_path)
                colorlist = [(random.randint(0, 210), random.randint(0, 210), random.randint(0, 210)) for i in
                             range(10000)]
                img_scene = vis_object_detection(img_scene, test_set, obj_cls, obj_boxes, obj_scores)
                cv2.imshow('sample', img_scene)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


        # cv2.imshow('sample',img_scene)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        result = {'objects': {
            'bbox': obj_boxes,
            'scores': obj_scores,
            'class': obj_cls, },
            'relationships': relationships
        }