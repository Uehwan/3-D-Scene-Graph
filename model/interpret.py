import sys
sys.path.append('./FactorizableNet')
from lib.fast_rcnn.nms_wrapper import nms
from lib.fast_rcnn.bbox_transform import bbox_transform_inv_hdn, clip_boxes
from lib.utils.nms import triplet_nms as triplet_nms_py
from sort.sort import Sort, iou
import numpy as np
from torch.autograd import Variable
import torchtext
import torch
from prior import relation_prior
from SGGenModel import VG_DR_NET_OBJ_IGNORES
from torch.nn.functional import cosine_similarity

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
    def __init__(self,args, data_set,ENABLE_TRACKING=None):
        self.tracker = Sort()
        self.args = args
        self.nms_thres = args.nms
        self.triplet_nms_thres =args.triplet_nms
        self.obj_thres = args.obj_thres
        self.triplet_thres = args.triplet_thres
        self.tobefiltered_objects = [26, 53, 134, 247, 179, 74, 226, 135, 145, 300, 253, 95, 11, 102,87]
        # 26: wheel, 53: backpack, 143:light, 247:camera, 179:board
        # 74:shoe, 226:chair, 135:shelf, 145:button, 300:cake, 253:knob, 95:wall, 11:door, 102:mirror,87:ceiling
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

        # Params for Statistics Based Scene Graph Inference
        self.relation_statistics = relation_prior.load_obj("model/prior/preprocessed/relation_prior_prob")
        self.joint_probability = relation_prior.load_obj("model/prior/preprocessed/object_prior_prob")
        self.spurious_rel_thres = 0.07
        self.rel_infer_thres = 0.9
        self.obj_infer_thres = 0.001
        self.data_set = data_set
        self.detected_obj_set = set()
        self.fasttext = torchtext.vocab.FastText()
        self.word_vecs, self.word_itos,self.word_stoi = self.prepare_wordvecs(num_vocabs=400,ignores=VG_DR_NET_OBJ_IGNORES)
        self.pred_stoi = {self.data_set.predicate_classes[i]: i for i in range(len(self.data_set.predicate_classes))}

    # p(x, y)
    def cal_p_xy_joint(self,x_ind, y_ind):
        p_xy = self.joint_probability[x_ind, y_ind] / np.sum(self.joint_probability)
        return p_xy

    # p(x|y)
    def cal_p_x_given_y(self,x_ind, y_ind):
        single_prob = np.sum(self.joint_probability, axis=1)
        p_y = single_prob[y_ind]
        p_xy = self.joint_probability[x_ind, y_ind]
        return p_xy / p_y

    # p(x|y,z) approximated
    def cal_p_x_given_yz(self,x_ind, y_ind, z_ind):
        p_x_given_y = self.cal_p_x_given_y(x_ind, y_ind)
        p_x_given_z = self.cal_p_x_given_y(x_ind, z_ind)
        return min(p_x_given_y, p_x_given_z)

    # True if p(x, z)^2 < p(x,y)*p(y,z)
    def check_prob_condition(self,x_ind,y_ind,z_ind):
        p_xz = self.cal_p_xy_joint(x_ind,z_ind)
        p_xy = self.cal_p_xy_joint(x_ind,y_ind)
        p_yz = self.cal_p_xy_joint(y_ind,z_ind)
        return p_xz**2 < p_xy*p_yz


    def prepare_wordvecs(self,num_vocabs = 400, ignores = VG_DR_NET_OBJ_IGNORES):
        word_inds = range(num_vocabs)
        word_inds = [x for x in word_inds if x not in ignores]
        word_txts = [self.data_set.object_classes[x] for x in word_inds]
        self.word_ind2vec = {ind:self.fasttext.vectors[self.fasttext.stoi[x]] for ind,x in zip(word_inds,word_txts)}

        word_vecs = torch.stack([self.fasttext.vectors[self.fasttext.stoi[x]] for x in word_txts]).cuda()
        word_itos = {i: self.data_set.object_classes[x] for i, x in enumerate(word_inds)}
        word_stoi = {self.data_set.object_classes[x]:i for i, x in enumerate(word_inds)}
        return word_vecs, word_itos, word_stoi


    def update_obj_set(self,obj_inds):
        for obj_ind in obj_inds[:,0]: self.detected_obj_set.add(obj_ind)


    def find_disconnected_pairs(self,obj_inds, relationships):
        connected_pairs = set(tuple(x) for x in relationships[:, :2].astype(int).tolist())
        disconnected_pairs = set()
        for i in range(len(obj_inds)):
            for j in range(len(obj_inds)):
                if i == j: continue
                if (i,j) in connected_pairs or (j,i) in connected_pairs: continue
                disconnected_pairs.add((i,j))
        return disconnected_pairs


    def missing_relation_inference(self,obj_inds,obj_boxes,disconnected_pairs):
        infered_relation=set()
        #print('discon:',disconnected_pairs)
        for i in range(len(disconnected_pairs)):
            pair = disconnected_pairs.pop()
            node1_box, node2_box = obj_boxes[pair[0]], obj_boxes[pair[1]]
            distance = self.distance_between_boxes(np.stack([node1_box, node2_box], axis=0))[0, 1]
            pair_txt = [self.data_set.object_classes[obj_inds[pair[0]][0]],
                        self.data_set.object_classes[obj_inds[pair[1]][0]]]
            candidate, prob, direction = relation_prior.most_probable_relation_for_unpaired(pair_txt, self.relation_statistics, int(distance))
            if candidate !=None and prob > self.rel_infer_thres:
                if not direction: pair = (pair[1],pair[0])
                infered_relation.add((pair[0],pair[1],self.pred_stoi[candidate],prob))
                pair_txt = [self.data_set.object_classes[obj_inds[pair[0]][0]],
                            self.data_set.object_classes[obj_inds[pair[1]][0]]]
                #print('dsfsfd:',pair_txt[0],pair_txt[1],candidate,prob)
        infered_relation= np.array(list(infered_relation)).reshape(-1, 4)
        #print(infered_relation)
        return infered_relation


    def missing_object_inference(self,obj_inds,disconnected_pairs):
        detected_obj_list = np.array(list(self.detected_obj_set))
        candidate_searchspace = [self.word_ind2vec[x] for x in detected_obj_list]
        candidate_searchspace = torch.stack(candidate_searchspace,dim=0).cuda()
        search_size = candidate_searchspace.shape[0]
        infered_obj_list = []

        for i in range(len(disconnected_pairs)):
            pair = disconnected_pairs.pop()
            ''' wordvec based candidate objects filtering '''
            #print(pair)
            sbj_vec = self.word_ind2vec[obj_inds[pair[0]][0]].cuda()
            obj_vec = self.word_ind2vec[obj_inds[pair[1]][0]].cuda()
            sim_sbj_obj = cosine_similarity(sbj_vec,obj_vec,dim=0)

            sbj_vec = sbj_vec.expand_as(candidate_searchspace)
            obj_vec = obj_vec.expand_as(candidate_searchspace)
            sim_cans_sbj = cosine_similarity(candidate_searchspace,sbj_vec, dim=1)
            sim_cans_obj = cosine_similarity(candidate_searchspace,obj_vec, dim=1)
            sim_sbj_obj = sim_sbj_obj.expand_as(sim_cans_obj)
            keep = (sim_cans_sbj + sim_cans_obj > 2 * sim_sbj_obj).nonzero().view(-1).cpu().numpy()
            #print(keep)
            #print(detected_obj_list)
            candidate_obj_list = detected_obj_list[keep]
            if len(candidate_obj_list) == 0: continue

            ''' statistics based candidate objects filtering '''
            keep=[]
            for i,obj_ind in enumerate(candidate_obj_list):
                if self.check_prob_condition(obj_inds[pair[0]][0],obj_ind,obj_inds[pair[1]][0]): keep.append(i)
            candidate_obj_list = candidate_obj_list[keep]
            if len(candidate_obj_list) == 0: continue

            ''' choose a candidate with best score above threshold'''
            probs = [self.cal_p_x_given_yz(candidate, obj_inds[pair[0]][0], obj_inds[pair[1]][0]) for candidate in candidate_obj_list]
            chosen_obj = candidate_obj_list[(np.array(probs)).argmax()]
            infered_obj_list.append(chosen_obj)
            #print(max(probs),self.data_set.object_classes[obj_inds[pair[0]][0]],
            #      self.data_set.object_classes[chosen_obj],
            #      self.data_set.object_classes[obj_inds[pair[1]][0]])


    def get_box_centers(self,boxes):
        # Define bounding box info
        center_x = (boxes[:, 0] + boxes[:, 2]) / 2
        center_y = (boxes[:, 1] + boxes[:, 3]) / 2
        centers = np.concatenate([center_x.reshape(-1, 1), center_y.reshape(-1, 1)], axis=1)
        return centers

    def distance_between_boxes(self,boxes):
        '''
        returns all possible distances between boxes

        :param boxes:
        :return: dist: distance between boxes[1] and boxes[2] ==> dist[1,2]
        '''
        centers = self.get_box_centers(boxes)
        centers_axis1 = np.repeat(centers,centers.shape[0],axis=0).reshape(-1,2)
        centers_axis2 = np.stack([centers for _ in range(centers.shape[0])]).reshape(-1, 2)
        dist = np.linalg.norm(centers_axis1 - centers_axis2, axis=1).reshape(-1,centers.shape[0])
        return dist


    def spurious_relation_rejection(self,obj_boxes,obj_cls,relationships):
        if self.args.disable_spurious: return range(len(relationships))
        subject_inds = obj_cls[relationships.astype(int)[:,0]][:, 0]
        pred_inds = relationships.astype(int)[:, 2]
        object_inds = obj_cls[relationships.astype(int)[:, 1]][:, 0]

        subject_boxes = obj_boxes[relationships.astype(int)[:,0]]
        object_boxes = obj_boxes[relationships.astype(int)[:,1]]

        keep = []
        for i, (sbj_ind, pred_ind, obj_ind, sbj_box, obj_box) in enumerate(zip(subject_inds,pred_inds,object_inds,
                                                                               subject_boxes,object_boxes)):
            relation_txt = [self.data_set.object_classes[sbj_ind],
                            self.data_set.predicate_classes[pred_ind],
                            self.data_set.object_classes[obj_ind]]
            distance = self.distance_between_boxes(np.stack([sbj_box,obj_box],axis=0))[0,1]
            prob = relation_prior.triplet_prob_from_statistics(relation_txt, self.relation_statistics, int(distance))
            print('prob: {prob:3.2f}     {sbj:15}{rel:15}{obj:15}'.format(prob=prob,
                                                                          sbj=relation_txt[0],
                                                                          rel=relation_txt[1],
                                                                          obj=relation_txt[2]))

            if prob > self.spurious_rel_thres: keep.append(i)

        return keep





    def interpret_graph(self,object_result, predicate_result,im_info):
        cls_prob_object, bbox_object, object_rois, reranked_score = object_result[:4]
        cls_prob_predicate, mat_phrase = predicate_result[:2]
        region_rois_num = predicate_result[2]

        obj_boxes, obj_scores, obj_cls, \
        subject_inds, object_inds, \
        subject_boxes, object_boxes, \
        subject_IDs, object_IDs, \
        predicate_inds, triplet_scores, relationships = \
            self.interpret_graph_(cls_prob_object, bbox_object, object_rois,
                                    cls_prob_predicate, mat_phrase, im_info,
                                    reranked_score)

        ''' missing object inference '''
        # self.update_obj_set(obj_cls)
        # disconnected_pairs = self.find_disconnected_pairs(obj_cls, relationships)
        # self.missing_object_inference(obj_cls,disconnected_pairs)
        ''' missing object infernce (end) '''

        ''' missing relation inference '''
        # infered_relations = self.missing_relation_inference(obj_cls,obj_boxes,disconnected_pairs)
        # print('size:',relationships.shape,infered_relations.shape)
        #
        # relationships = np.concatenate([relationships,infered_relations],axis=0)
        #
        # predicate_inds = relationships[:, 2].astype(int)
        # subject_boxes = obj_boxes[relationships[:, 0].astype(int)]
        # object_boxes = obj_boxes[relationships[:, 1].astype(int)]
        # subject_IDs = np.array([int(obj_boxes[int(relation[0])][4]) for relation in relationships])
        # object_IDs = np.array([int(obj_boxes[int(relation[1])][4]) for relation in relationships])
        # subject_inds = obj_cls[relationships[:, 0].astype(int)]
        # object_inds = obj_cls[relationships[:, 1].astype(int)]
        # subject_scores = [obj_scores[int(relation[0])] for relation in relationships]
        # pred_scores = [relation[3] / obj_scores[int(relation[0])] / obj_scores[int(relation[1])] for relation in
        #                relationships]
        # object_scores = [obj_scores[int(relation[1])] for relation in relationships]
        # triplet_scores = np.array(zip(subject_scores, pred_scores, object_scores))
        ''' missing relation inference (end) '''


        keep = self.spurious_relation_rejection(obj_boxes, obj_cls, relationships)

        return obj_boxes, obj_scores, obj_cls, \
               subject_inds[keep], object_inds[keep], \
               subject_boxes[keep], object_boxes[keep], \
               subject_IDs[keep], object_IDs[keep], \
               predicate_inds[keep], triplet_scores[keep], relationships[keep]



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
        # filter out objects with wrong class
        for i,ind in enumerate(inds):
            if ind[0] in self.tobefiltered_objects:
                scores[i].fill(0)


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
            # if keep_rel.size > 0:
            #     cutline_idx = max(keep_rel)
            #     relationships = relationships[:cutline_idx + 1]
            relationships = relationships[keep_rel]

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
        subject_IDs = np.array([int(obj_boxes[int(relation[0])][4]) for relation in relationships])
        object_IDs = np.array([int(obj_boxes[int(relation[1])][4]) for relation in relationships])


        subject_scores = [obj_scores[int(relation[0])] for relation in relationships]
        pred_scores = [relation[3] / obj_scores[int(relation[0])] / obj_scores[int(relation[1])] for relation in
                       relationships]
        object_scores = [obj_scores[int(relation[1])] for relation in relationships]
        triplet_scores = np.array(zip(subject_scores, pred_scores, object_scores))

        #print(relationships)


        return obj_boxes, obj_scores, obj_cls, \
               subject_inds, object_inds, \
               subject_boxes, object_boxes, \
               subject_IDs, object_IDs, \
               predicate_inds, triplet_scores, relationships