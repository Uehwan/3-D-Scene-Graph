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
from settings import parse_args
from pandas import Series, DataFrame
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
SCANNET_PATH = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/color/'
SCANNET_RESULT_PATH = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/object_detection/'
CAMPOSE_PATH = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/pose/'
DEPTH_PATH = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/depth/'
INTRINSIC_PATH = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/intrinsic/'

ETHP_PATH = './sort/mot_benchmark/train/ETH-Pedcross2/img1'
ETHS_PATH = './sort/mot_benchmark/train/ETH-Sunnyday/img1'
KITTI13_PATH = './sort/mot_benchmark/train/KITTI-13/img1'

#Load Camera intrinsic parameter
intrinsic_color = open(INTRINSIC_PATH + 'intrinsic_color.txt').read()
intrinsic_depth = open(INTRINSIC_PATH + 'intrinsic_depth.txt').read()
intrinsic_color = [item.split() for item in intrinsic_color.split('\n')[:-1]]
intrinsic_depth = [item.split() for item in intrinsic_depth.split('\n')[:-1]]
intrinsic_depth = np.matrix(intrinsic_depth, dtype='float')

# # Initiate ORB detector
# orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_HARRIS_SCORE)
# # Initiate Brute-Force Matcher for feature matching
# bf = cv2.BFMatcher(cv2.NORM_L2)
# prev_keypoints, prev_descriptors, prev_obj_inds, prev_color_IDs = [], [], [], []


# Initial Sort tracker
tracker = Sort()

imageFileList = sorted(os.listdir(SCANNET_PATH))
data_num = 0
for idx in range(len(imageFileList))[0:]:
    print('...........................................................................')
    print('Image '+str(idx))
    print('...........................................................................')
    ''' 1. Load an color/depth image and camera parameter'''
    if args.dataset == 'scannet':
        # Load an image from ScanNet Dataset
        img_path = osp.join(SCANNET_PATH, str(idx) + '.jpg')
        camera_pose = open(CAMPOSE_PATH + str(idx) + '.txt').read()
        depth_img = Image.open(DEPTH_PATH + str(idx) + '.png')
        ''' 1-2. Preprocess loaded camera parameter and depth info '''
        depth_pix = depth_img.load()
        pix_depth = []
        for ii in range(depth_img.size[0]):
            pix_row = []
            for jj in range(depth_img.size[1]):
                pix_row.append(depth_pix[ii, jj])
            pix_depth.append(pix_row)

        camera_pose = [item.split() for item in camera_pose.split('\n')[:-1]]

        p_matrix = [ intrinsic_color[0][:], intrinsic_color[1][:], intrinsic_color[2][:]]
        p_matrix = np.matrix(p_matrix, dtype='float')
        inv_p_matrix = np.linalg.pinv(p_matrix)

        R = np.matrix([camera_pose[0][0:3], camera_pose[1][0:3], camera_pose[2][0:3]], dtype='float')
        inv_R = np.linalg.inv(R)
        Trans = np.matrix([camera_pose[0][3], camera_pose[1][3], camera_pose[2][3]], dtype='float')
        sg = Digraph('structs', node_attr = {'shape': 'plaintext'})
    elif args.dataset == 'visual_genome':
        # Load an image from Visual Genome Dataset
        img_path = osp.join(cfg.IMG_DATA_DIR, test_set.annotations[idx]['path'])
    elif args.dataset == 'eth_sunnyday':
        idx+=1
        img_path = osp.join(ETHS_PATH, '%06d.jpg'%(idx))
    elif args.dataset == 'eth_pedcross2':
        idx += 1
        img_path = osp.join(ETHP_PATH, '%06d.jpg' % (idx))
    elif args.dataset == 'kitti13':
        idx += 1
        img_path = osp.join(KITTI13_PATH, '%06d.jpg' % (idx))

    else:
        raise NotImplementedError

    image_scene = cv2.imread(img_path)



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

    # ''' 4-2. Object Tracking (Same object Detection) '''
    # cropped_obj_imgs = []
    # keypoints, descriptors = [], []
    # same_node_list = np.zeros(obj_boxes.shape[0])
    # color_IDs = [random.randint(0,50) for _ in range(obj_boxes.shape[0])]
    # for i in range(obj_boxes.shape[0]):
    #      cropped_obj_img = image_scene[int(obj_boxes[i][1]):int(obj_boxes[i][3]) + 1, int(obj_boxes[i][0]):int(obj_boxes[i][2]) + 1]
    #      cropped_obj_imgs.append(cropped_obj_img)
    #      kp, des = orb.detectAndCompute(cropped_obj_img,None)
    #      cv2.drawKeypoints(cropped_obj_img,kp,cropped_obj_img)
    #      #cv2.imshow('cropped'+str(i),cropped_obj_img)
    #      keypoints.append(kp), descriptors.append(des)
    #      for j, prev_des in enumerate(prev_descriptors):
    #         if type(des) != np.ndarray or type(prev_des) != np.ndarray:
    #             match = None
    #         else:
    #             match_candidates = bf.match(des,prev_des)
    #             match = filter(lambda x: x.distance<250.0,match_candidates)
    #             print(test_set.object_classes[obj_inds[i]]+' '+str(len(match))+ '/' +str(len(match_candidates)))
    #             if float(len(match))/float(len(match_candidates)) > 0.1:
    #                 print('object '+str(i)+ ' and prev object ' +str(j)+' are the same!')
    #                 color_IDs[i] = prev_color_IDs[j]
    #                 same_node_list[i]=1
    # prev_keypoints, prev_descriptors, prev_obj_inds, prev_color_IDs = keypoints, descriptors, obj_inds, color_IDs
    # tracking_input = np.concatenate((obj_boxes, obj_scores.reshape(len(obj_scores), 1)), axis=1)
    # bboxes_and_uniqueIDs = tracker.update(tracking_input)
    # def filter_untracted(ref_bbox, tobefiltered_bbox, tobefiltered_inds):
    #     keep = []
    #     for bbox in ref_bbox:
    #         ious = [iou(bbox[:4], obj_box) for obj_box in tobefiltered_bbox]
    #         keep.append(np.argmax(ious))
    #     filtered_inds = tobefiltered_inds[keep]
    #     return keep
    # keep = filter_untracted(bboxes_and_uniqueIDs,obj_boxes,obj_inds)
    # obj_inds = obj_inds[keep]
    # keep = filter_untracted(bboxes_and_uniqueIDs,subject_boxes,subject_inds)
    # subject_inds, predicate_inds, object_inds = subject_inds[keep], predicate_inds[keep], object_inds[keep]
    # keep = filter_untracted(bboxes_and_uniqueIDs, object_boxes, object_inds)
    # subject_inds, predicate_inds, object_inds = subject_inds[keep], predicate_inds[keep], object_inds[keep]

    ''' 5. Print Scene Graph '''
    node_feature_list=[]
    if args.dataset == 'visual_genome':
        print('-ID--|----Object-----|---Score---------------------------------------------')
    else:   #scannet
        print('-ID--|----Object-----|---Score---|---3d_position (x, y, z)-----------------')
    for i, obj_ind in enumerate(obj_inds):
        if args.dataset == 'scannet':
            width = int(obj_boxes[i][2]) - int(obj_boxes[i][0])
            height = int(obj_boxes[i][3]) - int(obj_boxes[i][1])
            box_center_x = int(obj_boxes[i][0]) + width/2
            box_center_y = int(obj_boxes[i][1]) + height/2
            pose_2d = np.matrix([box_center_x, box_center_y, 1])
            pose_3d = pix_depth[box_center_x][box_center_y] * np.matmul(inv_p_matrix, pose_2d.transpose())
            pose_3d_world_coord = np.matmul(inv_R, pose_3d[0:3] - Trans.transpose())
            print('{obj_ID:5} {obj_cls:15} {obj_score:1.3f} {object_3d_pose_x:15.3f} {object_3d_pose_y:10.3f} {object_3d_pose_z:10.3f}'
                          .format(obj_ID=str(int(obj_boxes[i][4])), obj_cls=test_set.object_classes[obj_ind], obj_score=obj_scores[i],
                           object_3d_pose_x=pose_3d_world_coord.item(0), object_3d_pose_y=pose_3d_world_coord.item(1), object_3d_pose_z=pose_3d_world_coord.item(2) ))

            # node information in Dataframe
            node_feature_list.append([test_set.object_classes[obj_ind],
                                      str(int(obj_boxes[i][4])),
                                      [box_center_x, box_center_y, width, height],
                                      [round(pose_3d_world_coord.item(0),3), \
                                      round(pose_3d_world_coord.item(1),3), \
                                      round(pose_3d_world_coord.item(2),3)]])

        else:
            print('{obj_ID:5} {obj_cls:15} {obj_score:1.3f}'
                  .format(obj_ID=str(int(obj_boxes[i][4])), obj_cls=test_set.object_classes[obj_ind], obj_score=obj_scores[i]))

        cv2.rectangle(image_scene,
                      (int(obj_boxes[i][0]), int(obj_boxes[i][1])),
                      (int(obj_boxes[i][2]), int(obj_boxes[i][3])),
                      colorlist[int(obj_boxes[i][4])],
                      2)
        font_scale=0.5
        txt =str(int(obj_boxes[i][4])) + '. '+vis.get_class_string(obj_ind,obj_scores[i],test_set.object_classes)
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

    # Construct Dataframes of node feature
    rows = []
    if (data_num == 0):
        # node_feature : [name, obj_idx, bounding_box: [c_x, c_y, w, h], pose: [x,y,z], keypoints, descriptor]
        data = DataFrame({"node_feature": [node_feature_list]}, index= ["img"+str(idx)])
    else:
        rows.append(node_feature_list)
    relation_list = []

    print('-------Subject-------|------Predicate-----|--------Object---------|--Score-')
    for i in range(len(predicate_inds)):
        if predicate_inds[i] > 0: # predicate_inds[i] = 0 is the class 'irrelevant'
            print('{sbj_cls:9} {sbj_ID:4} {sbj_score:1.2f}  |  '
                  '{pred_cls:11} {pred_score:1.2f}  |  '
                  '{obj_cls:9} {obj_ID:4} {obj_score:1.2f}  |  '
                  '{triplet_score:1.3f}'.format(
                sbj_cls = test_set.object_classes[subject_inds[i]], sbj_score = triplet_scores[i][0],
                sbj_ID = str(subject_IDs[i]),
                pred_cls = test_set.predicate_classes[predicate_inds[i]] , pred_score = triplet_scores[i][1],
                obj_cls = test_set.object_classes[object_inds[i]], obj_score = triplet_scores[i][2],
                obj_ID = str(object_IDs[i]),
                triplet_score = np.prod(triplet_scores[i])))
            # node's Relation info in Dataframe
            relation_list.append([str(subject_IDs[i]), test_set.predicate_classes[predicate_inds[i]], str(object_IDs[i])])

    # Add relation info to Dataframe
    if (data_num == 0):
        # relation : [subj_id, pred_cls, obj_ID]
        data["relation"] = [relation_list]
        data_num +=1
    else:
        rows.append(relation_list)
        data.loc[len(data)] = rows
        data = data.rename(index={idx:"img"+str(idx)})

    print('.........................................................................')
    print(data.node_feature)
    '''
    data.node_feature[0] : image number select
    data.node_feature[0][0] : show one pack of object node info
    data.node_feature[0][0][0] : select node's info [0~6] (name, obj_idx, ...) 
    '''
    print('.........................................................................')
    print(data.relation) # similar format as node_feature
    
    
    for i in range(len(data.node_feature[idx])):
        sg.node('struct'+str(data.node_feature[idx][i][1]), '''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
          <TR>
            <TD PORT="'''+str(data.node_feature[idx][i][1])+'''" ROWSPAN="2">'''+str(data.node_feature[idx][i][0])+'''</TD>
            <TD COLSPAN="1">index</TD>
            <TD COLSPAN="3">position</TD>
          </TR>
          <TR>
            <TD>'''+str(data.node_feature[idx][i][1])+'''</TD>
            <TD>'''+str(data.node_feature[idx][i][3][0]) + '''</TD>
            <TD>'''+str(data.node_feature[idx][i][3][1])+'''</TD>
            <TD>'''+str(data.node_feature[idx][i][3][2])+'''</TD>
          </TR>
        </TABLE>>''')
    for j in range(len(data.relation[idx])):
        sg.edge('struct'+str(data.relation[idx][j][0])+':'+str(data.relation[idx][j][0]), 'struct'+str(data.relation[idx][j][2])+':'+str(data.relation[idx][j][2]), str(data.relation[idx][j][1]))
    sg.render('scene_graph_test/scene-graph_'+str(idx)+'.gv', view=True, cleanup=True)
    



    winname2 = 'image_scene'
    cv2.namedWindow(winname2)  # Create a named window
    cv2.moveWindow(winname2, 10, 10)
    cv2.imshow(winname2, image_scene)


    ''' 6. Print Captions '''
    print('Idx-|------Caption----------------------------------------------|--Score---')
    for i, caption in enumerate(region_caption):
        ans = [test_set.idx2word[caption_ind] for caption_ind in caption]
        if ans[0] != '<end>':
            sentence = ' '.join(c for c in ans if c != '<end>')
            print('{idx:5} {sentence:60} {score:1.3f}'.format(idx=str(i)+'.',sentence=sentence, score = region_scores[i]))
            cv2.rectangle(image_caption,
                          (int(region_boxes[i][0]),int(region_boxes[i][1])),
                          (int(region_boxes[i][2]),int(region_boxes[i][3])),
                          colorlist[i],
                          1)
            cv2.putText(image_caption,
                        str(i) + '. ' + sentence + ' ' + str(region_scores[i])[:4],
                        (int(region_boxes[i][0]),int(region_boxes[i][3])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colorlist[i],
                        1)
    winname1 = 'image_caption'
    #cv2.namedWindow(winname1)  # Create a named window
    #cv2.moveWindow(winname1, 10, 500)
    #cv2.imshow(winname1, image_caption)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(osp.join(SCANNET_RESULT_PATH, str(idx) + '.jpg'), image_scene)













