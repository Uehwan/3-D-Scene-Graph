import sys
sys.path.append('./FactorizableNet')
import random
import numpy.random as npr
import numpy as np
import yaml
from pprint import pprint
import cv2
import torch
from torch.autograd import Variable
from lib import network
import lib.datasets as datasets
import lib.utils.general_utils as utils
from model.settings import parse_args, testImageLoader
from PIL import Image
import os.path as osp
import os
from model import interpret, vis_tuning
from model.vis_tuning import tools_for_visualizing
from model.keyframe.keyframe_extracion import keyframe_checker
from model.SGGenModel import SGGen_MSDN, SGGen_DR_NET

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
interpreter = interpret.interpreter(args, test_set, ENABLE_TRACKING=False)
scene_graph = vis_tuning.scene_graph(args)
keyframe_extractor = keyframe_checker(args,
                                      thresh_key=args.thres_key,
                                      thresh_anchor=args.thres_anchor,
                                      max_group_len=args.max_group_len,
                                      intrinsic_depth=imgLoader.intrinsic_color,
                                      alpha=args.alpha,
                                      blurry_gain=args.gain,
                                      blurry_offset=args.offset,
                                      depth_shape=(480,640))

for idx in range(imgLoader.num_frames)[args.frame_start:args.frame_end]:
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
    IS_KEY_OR_ANCHOR, sharp_score, sharp_thres = keyframe_extractor.check_frame(image_scene, depth_img, camera_pose)
    winname_scene = '{idx:0004}. sharp_score: {score:05.1f}, sharp_thres: {thres:05.1f}, KEY_OR_ANCHOR: {flag:5}' \
        .format(idx=idx, score=sharp_score, thres=sharp_thres, flag=str(IS_KEY_OR_ANCHOR))
    print(winname_scene)
    image_original = image_scene.copy()
    if args.visualize:
        cv2.namedWindow('detection')  # Create a named window
        cv2.moveWindow('detection', 1400, 10)
        cv2.putText(image_original,
                    winname_scene,
                    (1, 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    # (150, 150, 150),
                    (30, 30, 200),
                    2)
        cv2.imshow('detection', image_original)
        cv2.waitKey(1)
    if not IS_KEY_OR_ANCHOR:
        continue

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


    ''' 5. Print 2D Object Detection '''
    # original image_scene
    img_obj_detected = tools_for_visualizing.vis_object_detection(image_scene.copy(), test_set, obj_cls[:, 0], obj_boxes, obj_scores[:, 0])

    if args.visualize:
        cv2.namedWindow('detection')  # Create a named window
        cv2.moveWindow('detection', 1400, 10)
        cv2.putText(img_obj_detected,
                    winname_scene,
                    (1, 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (30, 200, 30),
                    2)
        cv2.imshow('detection', image_original)

    if args.save_image:
        scene_name = args.scannet_path.split('/')[-1]
        try: os.makedirs(osp.join(args.vis_result_path, scene_name,'original'))
        except: pass
        try: os.makedirs(osp.join(args.vis_result_path, scene_name,'detection'))
        except: pass
        cv2.imwrite(osp.join(args.vis_result_path, scene_name,'original',str(idx) + '.jpg'), image_scene)
        cv2.imwrite(osp.join(args.vis_result_path, scene_name,'detection',str(idx) + '.jpg'), img_obj_detected)

    ''' 6. Merge Relations into 3D Scene Graph'''
    updated_image_scene = scene_graph.vis_scene_graph(image_scene.copy(), idx, test_set,
                                                      obj_cls, obj_boxes, obj_scores,
                                                      subject_inds, predicate_inds, object_inds,
                                                      subject_IDs, object_IDs, triplet_scores,relationships,
                                                      pix_depth, inv_p_matrix, inv_R, Trans, dataset=args.dataset)

    if args.visualize:
        cv2.namedWindow('updated')  # Create a named window
        cv2.moveWindow('updated', 1400, 520)
        cv2.putText(updated_image_scene,
                    winname_scene,
                    (1, 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (30, 200, 30),
                    2)
        cv2.imshow('updated', updated_image_scene)

    if args.save_image:
        scene_name = args.scannet_path.split('/')[-1]
        try: os.makedirs(osp.join(args.vis_result_path, scene_name,'updated'))
        except: pass
        cv2.imwrite(osp.join(args.vis_result_path, scene_name,'updated','updated'+str(idx) +'.jpg'), updated_image_scene)

    if args.visualize:
        cv2.waitKey(args.pause_time)











