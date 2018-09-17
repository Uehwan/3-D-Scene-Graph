import sys
sys.path.append('./FactorizableNet')
from models.HDN_v2.utils import interpret_relationships
from lib.fast_rcnn.nms_wrapper import nms
from lib import network
from lib.fast_rcnn.bbox_transform import bbox_transform_inv_hdn, clip_boxes
import lib.utils.logger as logger
from lib.utils.nms import triplet_nms as triplet_nms_py
from lib.utils.nms import unary_nms
from sort.sort import Sort, iou
import numpy as np
from torch.autograd import Variable


def filter_untracted(ref_bbox, tobefiltered_bbox):
    keep = []
    for bbox in ref_bbox:
        ious = [iou(bbox[:4], obj_box) for obj_box in tobefiltered_bbox]
        keep.append(np.argmax(ious))
    return keep

def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    keep = range(scores.shape[0])
    keep, scores, pred_boxes = zip(*sorted(zip(keep, scores, pred_boxes), key=lambda x: x[1][0])[::-1])
    keep, scores, pred_boxes = np.array(keep), np.array(scores), np.array(pred_boxes)
    dets = np.hstack((pred_boxes, scores[:,0][:, np.newaxis])).astype(np.float32)
    keep_keep = nms(dets, nms_thresh)
    keep_keep = keep_keep[:min(100, len(keep_keep))]
    keep = keep[keep_keep]
    if inds is None:
        return pred_boxes[keep_keep], scores[keep_keep], keep
    return pred_boxes[keep_keep], scores[keep_keep], inds[keep], keep


class interpreter(object):
    def __init__(self,args, ENABLE_TRACKING=None):
        self.tracker = Sort()
        self.args = args
        self.nms_thres = args.nms
        self.triplet_nms_thres =args.triplet_nms
        self.obj_thres = args.obj_thres
        self.triplet_thres = args.triplet_thres
        if ENABLE_TRACKING == None:
            self.ENABLE_TRACKING = False if self.args.dataset == 'visual_genome' else True
        else:
            self.ENABLE_TRACKING = ENABLE_TRACKING
        if self.ENABLE_TRACKING and self.args.path_opt.split('/')[-1] == 'VG-DR-Net.yaml':
            self.tobefiltered_predicates = [0,6,10,18,19,20,22,23,24]
            # 0:backgrounds, 6:eat,10:wear, 18:ride, 19:watch, 20:play, 22:enjoy, 23:read, 24:cut
        elif self.ENABLE_TRACKING and self.args.path_opt.split('/')[-1] == 'VG-MSDN.yaml':
            self.tobefiltered_predicates = [12, 18, 27, 28, 30, 31, 32, 35]
        else:
            self.tobefiltered_predicates = []

    def interpret_graph(self,object_result, predicate_result,im_info):
        cls_prob_object, bbox_object, object_rois, reranked_score = object_result[:4]
        cls_prob_predicate, mat_phrase = predicate_result[:2]
        region_rois_num = predicate_result[2]
        return self.interpret_graph_(cls_prob_object, bbox_object, object_rois,
                                cls_prob_predicate, mat_phrase, im_info,
                                reranked_score)




    def interpret_graph_(self,cls_prob_object, bbox_object, object_rois,
                                cls_prob_predicate, mat_phrase, im_info,
                                reranked_score=None):

        obj_boxes, obj_scores, obj_cls, subject_inds, object_inds, \
        subject_boxes, object_boxes, predicate_inds, \
        sub_assignment, obj_assignment, total_score = \
            self.interpret_relationships(cls_prob_object, bbox_object, object_rois,
                                         cls_prob_predicate, mat_phrase, im_info,
                                         nms=self.nms_thres, topk_pred=2, topk_obj=3,
                                         use_gt_boxes=False,
                                         triplet_nms=self.triplet_nms_thres,
                                         reranked_score=reranked_score)

        obj_boxes, obj_scores, obj_cls, \
        subject_inds, object_inds, \
        subject_boxes, object_boxes, \
        subject_IDs, object_IDs, \
        predicate_inds, triplet_scores, relationships = self.filter_and_tracking(obj_boxes, obj_scores, obj_cls,
                                                                                     subject_inds, object_inds,
                                                                                     subject_boxes, object_boxes,
                                                                                     predicate_inds,
                                                                                     sub_assignment, obj_assignment,
                                                                                     total_score)

        return obj_boxes, obj_scores, obj_cls, \
        subject_inds, object_inds, \
        subject_boxes, object_boxes, \
        subject_IDs, object_IDs, \
        predicate_inds, triplet_scores, relationships

    def interpret_relationships(self, cls_prob, bbox_pred, rois, cls_prob_predicate,
                                mat_phrase, im_info, nms=-1., clip=True, min_score=0.01,
                                top_N=100, use_gt_boxes=False, triplet_nms=-1., topk_pred=2,topk_obj=3,
                                reranked_score=None):

        scores, inds = cls_prob[:, 1:].data.topk(k=topk_obj,dim=1)
        if reranked_score is not None:
            if isinstance(reranked_score, Variable):
                reranked_score = reranked_score.data
            scores *= reranked_score
        inds += 1
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        predicate_scores, predicate_inds = cls_prob_predicate[:, 1:].data.topk(dim=1, k=topk_pred)
        predicate_inds += 1
        predicate_scores, predicate_inds = predicate_scores.cpu().numpy().reshape(
            -1), predicate_inds.cpu().numpy().reshape(-1)

        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        box_deltas = np.asarray([
            box_deltas[i, (inds[i][0] * 4): (inds[i][0] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        keep = range(scores.shape[0])
        if use_gt_boxes:
            triplet_nms = -1.
            pred_boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
        else:
            pred_boxes = bbox_transform_inv_hdn(rois.data.cpu().numpy()[:, 1:5], box_deltas) / im_info[0][2]
            pred_boxes = clip_boxes(pred_boxes, im_info[0][:2] / im_info[0][2])

            # nms
            if nms > 0. and pred_boxes.shape[0] > 0:
                assert nms < 1., 'Wrong nms parameters'
                pred_boxes, scores, inds, keep = nms_detections(pred_boxes, scores, nms, inds=inds)

        sub_list = np.array([], dtype=int)
        obj_list = np.array([], dtype=int)
        pred_list = np.array([], dtype=int)

        # mapping the object id
        mapping = np.ones(cls_prob.size(0), dtype=np.int64) * -1
        mapping[keep] = range(len(keep))

        sub_list = mapping[mat_phrase[:, 0]]
        obj_list = mapping[mat_phrase[:, 1]]
        pred_remain = np.logical_and(sub_list >= 0, obj_list >= 0)
        pred_list = np.where(pred_remain)[0]
        sub_list = sub_list[pred_remain]
        obj_list = obj_list[pred_remain]

        # expand the sub/obj and pred list to k-column
        pred_list = np.vstack([pred_list * topk_pred + i for i in range(topk_pred)]).transpose().reshape(-1)
        sub_list = np.vstack([sub_list for i in range(topk_pred)]).transpose().reshape(-1)
        obj_list = np.vstack([obj_list for i in range(topk_pred)]).transpose().reshape(-1)

        if use_gt_boxes:
            total_scores = predicate_scores[pred_list]
        else:
            total_scores = predicate_scores[pred_list] * scores[sub_list][:,0] * scores[obj_list][:,0]

        top_N_list = total_scores.argsort()[::-1][:10000]
        total_scores = total_scores[top_N_list]
        pred_ids = predicate_inds[pred_list[top_N_list]]  # category of predicates
        sub_assignment = sub_list[top_N_list]  # subjects assignments
        obj_assignment = obj_list[top_N_list]  # objects assignments
        sub_ids = inds[:,0][sub_assignment]  # category of subjects
        obj_ids = inds[:,0][obj_assignment]  # category of objects
        sub_boxes = pred_boxes[sub_assignment]  # boxes of subjects
        obj_boxes = pred_boxes[obj_assignment]  # boxes of objects

        if triplet_nms > 0.:
            sub_ids, obj_ids, pred_ids, sub_boxes, obj_boxes, keep = triplet_nms_py(sub_ids, obj_ids, pred_ids,
                                                                                    sub_boxes, obj_boxes, triplet_nms)
            sub_assignment = sub_assignment[keep]
            obj_assignment = obj_assignment[keep]
            total_scores = total_scores[keep]
        if len(sub_list) == 0:
            print('No Relatinoship remains')
            # pdb.set_trace()

        return pred_boxes, scores, inds, sub_ids, obj_ids, sub_boxes, obj_boxes, pred_ids, sub_assignment, obj_assignment, total_scores

    def filter_and_tracking(self, obj_boxes, obj_scores, obj_cls,
                            subject_inds, object_inds,
                            subject_boxes, object_boxes, predicate_inds,
                            sub_assignment, obj_assignment, total_score):

        relationships = np.array(zip(sub_assignment, obj_assignment, predicate_inds, total_score))

        # filter out bboxes who has low obj_score
        keep_obj = np.where(obj_scores[:,0] >= self.obj_thres)[0]
        if keep_obj.size == 0:
            print("no object detected ...")
            keep_obj= [0]
        cutline_idx = max(keep_obj)
        obj_scores = obj_scores[:cutline_idx + 1]
        obj_boxes = obj_boxes[:cutline_idx + 1]
        obj_cls = obj_cls[:cutline_idx + 1]

        # filter out triplets whose obj/sbj have low obj_score
        if relationships.size > 0:
            keep_sub_assign = np.where(relationships[:, 0] <= cutline_idx)[0]
            relationships = relationships[keep_sub_assign]
        if relationships.size > 0:
            keep_obj_assign = np.where(relationships[:, 1] <= cutline_idx)[0]
            relationships = relationships[keep_obj_assign]

        # filter out triplets who have low total_score
        if relationships.size > 0:
            keep_rel = np.where(relationships[:, 3] >= self.triplet_thres)[0]  # MSDN:0.02, DR-NET:0.03
            if keep_rel.size > 0:
                cutline_idx = max(keep_rel)
                relationships = relationships[:cutline_idx + 1]

        # filter out triplets whose sub equal obj
        if relationships.size > 0:

            #keep_rel = np.where(relationships[:, 0] != relationships[:, 1])[0]
            #relationships = relationships[keep_rel]
            keep_rel = []
            for i,relation in enumerate(relationships):
                if relation[0] != relation[1]:
                    keep_rel.append(i)
            keep_rel = np.array(keep_rel).astype(int)
            relationships = relationships[keep_rel]
            # print('filter1')
            # print(relationships.astype(int))



        # filter out triplets whose predicate is related to human behavior.
        if relationships.size > 0:
            keep_rel = []
            for i,relation in enumerate(relationships):
                if int(relation[2]) not in self.tobefiltered_predicates:
                    keep_rel.append(i)
            keep_rel = np.array(keep_rel).astype(int)
            #print('keep_rel:',keep_rel)
            relationships = relationships[keep_rel]
            # print('filter2')
            # print(relationships.astype(int))

        # Object tracking
        # Filter out all un-tracked objects and triplets
        if self.ENABLE_TRACKING:
            print(obj_boxes.shape)
            tracking_input = np.concatenate((obj_boxes, obj_scores[:,0].reshape(len(obj_scores), 1)), axis=1)
            bboxes_and_uniqueIDs = self.tracker.update(tracking_input)
            keep = filter_untracted(bboxes_and_uniqueIDs, obj_boxes)
            print(relationships.shape)

            # filter out triplets whose obj/sbj is untracked.
            if relationships.size >0:
                keep_sub_assign = [np.where(relationships[:, 0] == keep_idx) for keep_idx in keep]
                if len(keep_sub_assign) > 0:
                    keep_sub_assign = np.concatenate(keep_sub_assign, axis=1).flatten()
                    relationships = relationships[keep_sub_assign]
                else:
                    relationships = relationships[np.array([]).astype(int)]
            if relationships.size > 0:
                keep_obj_assign = [np.where(relationships[:, 1] == keep_idx) for keep_idx in keep]
                if len(keep_obj_assign) > 0:
                    keep_obj_assign = np.concatenate(keep_obj_assign, axis=1).flatten()
                    relationships = relationships[keep_obj_assign]
                else:
                    relationships = relationships[np.array([]).astype(int)]
            #
            print('filter3')
            print(relationships.astype(int))
            print(keep)
            rel = relationships.copy()
            for i, k in enumerate(keep):
                relationships[:,:2][rel[:,:2] == k] = i



            sorted = relationships[:,3].argsort()[::-1]
            relationships = relationships[sorted]
            #print('filter4')
            #print(relationships[:,3])

            subject_inds = obj_cls[relationships[:, 0].astype(int)]
            object_inds = obj_cls[relationships[:, 1].astype(int)]


            obj_boxes = np.concatenate([obj_boxes, np.zeros([obj_boxes.shape[0], 1])], axis=1)
            for i, keep_idx in enumerate(keep):
                obj_boxes[keep_idx] = bboxes_and_uniqueIDs[i]
            obj_scores = obj_scores[keep]
            obj_cls = obj_cls[keep]
            obj_boxes = obj_boxes[keep]

            #obj_boxes = bboxes_and_uniqueIDs

            print(obj_scores.shape)
            print(obj_cls.shape)
            print(obj_boxes.shape)
            print(relationships.shape)

        else:
            obj_boxes = np.concatenate([obj_boxes, np.zeros([obj_boxes.shape[0], 1])], axis=1)
            for i in range(len(obj_boxes)):
                obj_boxes[i][4] = i
            subject_inds = obj_cls[relationships[:, 0].astype(int)]
            object_inds = obj_cls[relationships[:, 1].astype(int)]
            #subject_boxes = obj_boxes[relationships[:, 0].astype(int)]
            #object_boxes = obj_boxes[relationships[:, 1].astype(int)]
            #subject_IDs = subject_boxes[:, 4].astype(int)
            #object_IDs = object_boxes[:, 4].astype(int)


        predicate_inds = relationships[:, 2].astype(int)
        subject_boxes = obj_boxes[relationships[:, 0].astype(int)]
        object_boxes = obj_boxes[relationships[:, 1].astype(int)]
        subject_IDs = [int(obj_boxes[int(relation[0])][4]) for relation in relationships]
        object_IDs = [int(obj_boxes[int(relation[1])][4]) for relation in relationships]


        subject_scores = [obj_scores[int(relation[0])] for relation in relationships]
        pred_scores = [relation[3] / obj_scores[int(relation[0])] / obj_scores[int(relation[1])] for relation in
                       relationships]
        object_scores = [obj_scores[int(relation[1])] for relation in relationships]
        triplet_scores = zip(subject_scores, pred_scores, object_scores)

        return obj_boxes, obj_scores, obj_cls, \
               subject_inds, object_inds, \
               subject_boxes, object_boxes, \
               subject_IDs, object_IDs, \
               predicate_inds, triplet_scores, relationships