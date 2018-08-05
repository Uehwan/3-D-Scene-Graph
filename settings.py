import argparse
from faster_rcnn.fast_rcnn.config import cfg
import numpy as np
import os.path as osp
import os
from PIL import Image
import cv2

def parse_args():
    parser = argparse.ArgumentParser('Options for testing Hierarchical Descriptive Model in pytorch')
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
    parser.add_argument('--dataset' ,type=str, default='scannet',
                        help='choose a dataset, "visual_genome", "scannet","eth_pedcross2", "eth_sunnyday"')
    parser.add_argument('--scannet_img_path', type=str,
                        default='/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/color/')
    parser.add_argument('--mot_benchmark_path', type=str,
                        default='./sort/mot_benchmark/')
    parser.add_argument('--vis_result_path',type=str,default='./vis_result')
    parser.add_argument('--top_N_triplets', type=int, default=10,
                        help='Only top N triplets are selected in descending order of score.')
    parser.add_argument('--top_N_captions', type=int, default=5,
                        help='Only top N captions are selected in descending order of score')
    args = parser.parse_args()

    return args


class testImageLoader(object):
    def __init__(self,args):
        self.args = args
        self.scannet_img_path = args.scannet_img_path
        self.scannet_depth_path = osp.join(args.scannet_img_path,'..','depth')
        self.scannet_intrinsic_path = osp.join(args.scannet_img_path,'..','intrinsic')
        print(self.scannet_intrinsic_path)
        self.scannet_pose_path = osp.join(args.scannet_img_path,'..','pose')
        # Load Camera intrinsic parameter
        self.intrinsic_color = open(osp.join(self.scannet_intrinsic_path, 'intrinsic_color.txt')).read()
        self.intrinsic_depth = open(osp.join(self.scannet_intrinsic_path, 'intrinsic_depth.txt')).read()
        self.intrinsic_color = [item.split() for item in self.intrinsic_color.split('\n')[:-1]]
        self.intrinsic_depth = [item.split() for item in self.intrinsic_depth.split('\n')[:-1]]
        self.intrinsic_depth = np.matrix(self.intrinsic_depth, dtype='float')

        self.mot_benchmark_train = ['ADL-Rundle-6', 'ADL-Rundle-8','ETH-Bahnhof','ETH-Pedcross2','ETH-Sunnyday',
                               'KITTI-13', 'KITTI-17','PETS09-S2L1','TUD-Campus','TUD-Stadtmitte','Venice-2']
        self.mot_benchmark_test = ['ADL-Rundle-1', 'ADL-Rundle-3','AVG-TownCentre','ETH-Crossing','ETH-Jelmoli',
                               'ETH-Linthescher', 'KITTI-16', 'KITTI-19','PETS09-S2L2','TUD-Crossing','Venice-1']
        if self.args.dataset == 'scannet':
            self.img_folder_path = self.scannet_img_path
        elif self.args.dataset == 'visual_genome':
            self.img_folder_path = cfg.IMG_DATA_DIR
        elif self.args.dataset in self.mot_benchmark_train:
            mot_train_path = osp.join(args.mot_benchmark_path, 'train')
            self.img_folder_path = osp.join(mot_train_path,self.args.dataset,'img1')
        elif self.args.dataset in self.mot_benchmark_test:
            mot_test_path = osp.join(args.mot_benchmark_path, 'test')
            self.img_folder_path = osp.join(mot_test_path, self.args.dataset,'img1')
        else:
            raise NotImplementedError

        self.num_frames = len(os.listdir(self.img_folder_path))

    def gen_img_path(self,frame_idx):

        if self.args.dataset == 'scannet':
            # Load an image from ScanNet Dataset
            img_path = osp.join(self.img_folder_path, str(frame_idx) + '.jpg')
            camera_pose = open(osp.join(self.scannet_pose_path, str(frame_idx) + '.txt')).read()
            depth_img = Image.open(osp.join(self.scannet_depth_path,str(frame_idx) + '.png'))
            # Preprocess loaded camera parameter and depth info
            depth_pix = depth_img.load()
            pix_depth = []
            for ii in range(depth_img.size[0]):
                pix_row = []
                for jj in range(depth_img.size[1]):
                    pix_row.append(depth_pix[ii, jj])
                pix_depth.append(pix_row)

            camera_pose = [item.split() for item in camera_pose.split('\n')[:-1]]
            p_matrix = [ self.intrinsic_color[0][:], self.intrinsic_color[1][:], self.intrinsic_color[2][:]]
            p_matrix = np.matrix(p_matrix, dtype='float')
            inv_p_matrix = np.linalg.pinv(p_matrix)
            R = np.matrix([camera_pose[0][0:3], camera_pose[1][0:3], camera_pose[2][0:3]], dtype='float')
            inv_R = np.linalg.inv(R)
            Trans = np.matrix([camera_pose[0][3], camera_pose[1][3], camera_pose[2][3]], dtype='float')
            img_scene = cv2.imread(img_path)
            return img_scene, depth_img, pix_depth, inv_p_matrix, inv_R, Trans

        elif self.args.dataset == 'visual_genome':
            # Load an image from Visual Genome Dataset
            frame_idx += 1
            img_path = osp.join(self.img_folder_path, str(frame_idx)+'.jpg')
        elif self.args.dataset in self.mot_benchmark_train or self.args.dataset in self.mot_benchmark_test:
            frame_idx += 1
            img_path = osp.join(self.img_folder_path, '%06d.jpg' % (frame_idx))
        else:
            raise NotImplementedError
        img_scene = cv2.imread(img_path)
        return img_scene
