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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import rgb2hex
import time
import matplotlib.animation as animation

TIME_IT = cfg.TIME_IT
parser = argparse.ArgumentParser('Options for training Hierarchical Descriptive Model in pytorch')
# Training parameters
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='base learning rate for training')
parser.add_argument('--max_epoch', type=int, default=10, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=1000, help='Interval for Logging')
parser.add_argument('--step_size', type=int, default = 2, help='Step size for reduce learning rate')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
parser.add_argument('--load_RPN', action='store_true', help='To end-to-end train from the scratch')
parser.add_argument('--enable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')
# structure settings
parser.add_argument('--disable_language_model', action='store_true', help='To disable the Lanuage Model ')
parser.add_argument('--mps_feature_len', type=int, default=1024, help='The expected feature length of message passing')
parser.add_argument('--dropout', action='store_true', help='To enables the dropout')
parser.add_argument('--MPS_iter', type=int, default=1, help='Iterations for Message Passing')
parser.add_argument('--gate_width', type=int, default=128, help='The number filters for gate functions in GRU')
parser.add_argument('--nhidden_caption', type=int, default=512, help='The size of hidden feature in language model')
parser.add_argument('--nembedding', type=int, default=256, help='The size of word embedding')
parser.add_argument('--rnn_type', type=str, default='LSTM_baseline', help='Select the architecture of RNN in caption model[LSTM_im | LSTM_normal]')
parser.add_argument('--caption_use_bias', action='store_true', help='Use the flap to enable the bias term to caption model')
parser.add_argument('--caption_use_dropout', action='store_const', const=0.5, default=0., help='Set to use dropout in caption model')
parser.add_argument('--enable_bbox_reg', dest='region_bbox_reg', action='store_true')
parser.add_argument('--disable_bbox_reg', dest='region_bbox_reg', action='store_false')
parser.set_defaults(region_bbox_reg=True)
parser.add_argument('--use_kernel_function', action='store_true')
# Environment Settings
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--saved_model_path', type=str, default = 'model/pretrained_models/VGG_imagenet.npy', help='The Model used for initialize')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/HDN', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='HDN', help='The name for saving model.')
parser.add_argument('--nesterov', action='store_true', help='Set to use the nesterov for SGD')
parser.add_argument('--finetune_language_model', action='store_true', help='Set to disable the update of other parameters')
parser.add_argument('--optimizer', type=int, default=0, help='which optimizer used for optimize language model [0: SGD | 1: Adam | 2: Adagrad]')
# Demo Settings by jmpark
parser.add_argument('--dataset' ,type=str, default='visual_genome', help='choose a dataset, "visual_genome" or  "scannet"')
parser.add_argument('--top_N_triplets', type=int, default=10, help='Only top N triplets are selected in descending order of score.')
parser.add_argument('--top_N_captions', type=int, default=5, help='Only top N captions are selected in descending order of score')

args = parser.parse_args()
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
colorlist = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for i in range(100)]

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
# Sample random scales to use for each image in this batch
SCANNET_PWD = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/color'
RESULT_PWD = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/object_detection'
CAMPOSE_PWD = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/pose/'
DEPTH_PWD = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/depth/'
INTRINSIC_PWD = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/intrinsic/'

#Load Camera intrinsic parameter
intrinsic_color = open(INTRINSIC_PWD + 'intrinsic_color.txt').read()
intrinsic_depth = open(INTRINSIC_PWD + 'intrinsic_depth.txt').read()
intrinsic_color = [item.split() for item in intrinsic_color.split('\n')[:-1]]
intrinsic_depth = [item.split() for item in intrinsic_depth.split('\n')[:-1]]
intrinsic_depth = np.matrix(intrinsic_depth, dtype='float')

imageFileList = sorted(os.listdir(SCANNET_PWD))
if args.dataset == 'scannet':
    X = []
    Y = []
    Z = []
    color = []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

#def animate(num, datalines, lines) :
    




for idx in range(len(imageFileList)):
    print('.........................................................................')
    print('Image '+str(idx))
    print('.........................................................................')
    ''' 1. Load an color/depth image and camera parameter'''
    if args.dataset == 'scannet':
        # Load an image from ScanNet Dataset
        img_path = osp.join(SCANNET_PWD, str(idx)+'.jpg')
        camera_pose = open(CAMPOSE_PWD + str(idx) + '.txt').read()
        depth_img = Image.open(DEPTH_PWD + str(idx) + '.png')
        ''' 1-2. Preprocessing Loaded camera parameter and depth info '''
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
    elif args.dataset == 'visual_genome':
        # Load an image from Visual Genome Dataset
        img_path = osp.join(cfg.IMG_DATA_DIR, test_set.annotations[idx]['path'])
    else:
        raise NotImplementedError

    image_scene = cv2.imread(img_path)

    ''' 2. Rescale & Normalization '''
    # Resize the image to target scale
    if args.dataset == 'scannet':
        image_scene= cv2.resize(image_scene, (depth_img.size), interpolation= cv2.INTER_CUBIC)
        image_caption = image_scene.copy()
        cv2.imshow('image_scene', image_scene)
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
    object_result, predicate_result, region_result = net(im_data.unsqueeze(0),im_info,graph_generation=True)
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
    if args.dataset == 'visual_genome':
        print('Idx-|----Object-----|---Score--------------------------------------------')
    else:   #scannet
        print('Idx-|--_-Object-----|---Score---|---3d_position (x, y, z)----------------')
    for i, obj_ind in enumerate(obj_inds):
        if args.dataset == 'scannet':
            width = int(obj_boxes[i][2]) - int(obj_boxes[i][0])
            height = int(obj_boxes[i][3]) - int(obj_boxes[i][1])
            box_center_x = int(obj_boxes[i][0]) + int(width/2)
            box_center_y = int(obj_boxes[i][1]) + int(height/2)
            box_center_color = image_scene[box_center_y][box_center_x]
            
            pose_2d = np.matrix([box_center_x, box_center_y, 1])
            pose_3d = pix_depth[box_center_x][box_center_y] * np.matmul(inv_p_matrix, pose_2d.transpose()) 
            pose_3d_world_coord = np.matmul(inv_R, pose_3d[0:3] - Trans.transpose())

            X.append(pose_3d_world_coord.item(0))
            Y.append(pose_3d_world_coord.item(1))
            Z.append(pose_3d_world_coord.item(2))
            color.append([box_center_color[0]/255.0, box_center_color[1]/255.0, box_center_color[2]/255.0])
            ax.scatter(X,Y,Z, c= color, marker='.')
            print('{obj_ID:5} {obj_cls:15} {obj_score:1.3f} {object_3d_pose_x:15.3f} {object_3d_pose_y:10.3f} {object_3d_pose_z:10.3f}'.format(obj_ID=str(i), obj_cls=test_set.object_classes[obj_ind], obj_score=obj_scores[i], object_3d_pose_x=pose_3d_world_coord.item(0),object_3d_pose_y=pose_3d_world_coord.item(1), object_3d_pose_z=pose_3d_world_coord.item(2) ))

        else:    
            print('{obj_ID:5} {obj_cls:15} {obj_score:1.3f}'
                  .format(obj_ID=str(i), obj_cls=test_set.object_classes[obj_ind], obj_score=obj_scores[i]))
        cv2.rectangle(image_scene,
                     (int(obj_boxes[i][0]),int(obj_boxes[i][1])),
                     (int(obj_boxes[i][2]),int(obj_boxes[i][3])),
                     colorlist[i],
                     1)
        cv2.putText(image_scene,
                str(i)+'. '+str(test_set.object_classes[obj_ind]) + ' ' + str(obj_scores[i])[:5],
                (int(obj_boxes[i][0]),int(obj_boxes[i][3])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colorlist[i],
                1)

    print('------Subject-------|------Predicate-----|--------Object--------|--Score-')
    for i in range(len(predicate_inds)):
        if predicate_inds[i] > 0: # predicate_inds[i] = 0 is the class 'irrelevant'
            print('{sbj_cls:9} {sbj_ID:3} {sbj_score:1.2f}  |  '
                  '{pred_cls:11} {pred_score:1.2f}  |  '
                  '{obj_cls:9} {obj_ID:3} {obj_score:1.2f}  |  '
                  '{triplet_score:1.3f}'.format(
                sbj_cls = test_set.object_classes[subject_inds[i]], sbj_score = triplet_scores[i][0],
                sbj_ID = str(subject_IDs[i]),
                pred_cls = test_set.predicate_classes[predicate_inds[i]] , pred_score = triplet_scores[i][1],
                obj_cls = test_set.object_classes[object_inds[i]], obj_score = triplet_scores[i][2],
                obj_ID = str(object_IDs[i]),
                triplet_score = np.prod(triplet_scores[i])))

    winname2 = 'image_scene'
    cv2.namedWindow(winname2)  # Create a named window
    cv2.moveWindow(winname2, 10, 10)
    cv2.imshow(winname2, image_scene)


    ''' 6. Print Captions '''
    print('Idx-|------Caption----------------------------------------------|--Score-')
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
    cv2.namedWindow(winname1)  # Create a named window
    cv2.moveWindow(winname1, 10, 500)
    cv2.imshow(winname1, image_caption)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.imwrite(osp.join(RESULT_PWD, str(idx)+'.jpg'),image_scene)

if args.dataset == 'scannet':
    ani = animation.FuncAnimation(fig, animate, 1, interval=10000/3)
    plt.show()










