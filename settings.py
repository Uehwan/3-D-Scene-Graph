import sys
sys.path.append('./FactorizableNet')
import argparse
import numpy as np
import os.path as osp
import os
from PIL import Image
import cv2
import yaml
import lib.utils.general_utils as utils

def parse_args():
    parser = argparse.ArgumentParser('Options for Running 3D-Scene-Graph-Generator in pytorch')
    parser.add_argument('--pretrained_model', type=str,
                        default='FactorizableNet/output/trained_models/Model-VG-DR-Net.h5',
                        help='path to pretrained_model, Model-VG-DR-Net.h5 or Model-VG-MSDN.h5')
    parser.add_argument('--path_opt', default='options/models/VG-DR-Net.yaml', type=str,
                        help='path to a yaml options file, VG-DR-Net.yaml or VG-MSDN.yaml')
    parser.add_argument('--dataset_option', type=str, default='normal',
                        help='data split selection [small | fat | normal]')
    parser.add_argument('--batch_size', type=int, help='#images per batch')
    parser.add_argument('--workers', type=int, default=4, help='#idataloader workers')
    # model init

    # Environment Settings
    parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
    parser.add_argument('--nms', type=float, default=0.2,
                        help='NMS threshold for post object NMS (negative means not NMS)')
    parser.add_argument('--triplet_nms', type=float, default=0.4,
                        help='Triplet NMS threshold for post object NMS (negative means not NMS)')
    # testing settings
    parser.add_argument('--use_gt_boxes', action='store_true', help='Use ground truth bounding boxes for evaluation')
    # Demo Settings by jmpark
    parser.add_argument('--dataset' ,type=str, default='scannet',
                        help='choose a dataset. Example: "visual_genome", "scannet","ETH-Pedcross2", "ETH-Sunnyday"')
    parser.add_argument('--scannet_path', type=str,
                        default='./data/scene0507/', help='scene0507')
    parser.add_argument('--mot_benchmark_path', type=str,
                        default='./data/mot_benchmark/')
    parser.add_argument('--vis_result_path',type=str,default='./vis_result')
    parser.add_argument('--obj_thres', type=float, default=0.25,
                        help='object recognition threshold score')
    parser.add_argument('--triplet_thres', type=float, default=0.1,
                        help='Triplet recognition threshold score ')
    args = parser.parse_args()

    return args


class testImageLoader(object):
    def __init__(self,args):
        self.args = args
        self.mot_benchmark_train = ['ADL-Rundle-6', 'ADL-Rundle-8','ETH-Bahnhof','ETH-Pedcross2','ETH-Sunnyday',
                               'KITTI-13', 'KITTI-17','PETS09-S2L1','TUD-Campus','TUD-Stadtmitte','Venice-2']
        self.mot_benchmark_test = ['ADL-Rundle-1', 'ADL-Rundle-3','AVG-TownCentre','ETH-Crossing','ETH-Jelmoli',
                               'ETH-Linthescher', 'KITTI-16', 'KITTI-19','PETS09-S2L2','TUD-Crossing','Venice-1']
        if self.args.dataset == 'scannet':
            self.scannet_img_path = osp.join(args.scannet_path, 'color')
            self.scannet_depth_path = osp.join(args.scannet_path, 'depth')
            self.scannet_intrinsic_path = osp.join(args.scannet_path, 'intrinsic')
            self.scannet_pose_path = osp.join(args.scannet_path, 'pose')
            # Load Camera intrinsic parameter
            self.intrinsic_color = open(osp.join(self.scannet_intrinsic_path, 'intrinsic_color.txt')).read()
            self.intrinsic_depth = open(osp.join(self.scannet_intrinsic_path, 'intrinsic_depth.txt')).read()
            self.intrinsic_color = [item.split() for item in self.intrinsic_color.split('\n')[:-1]]
            self.intrinsic_depth = [item.split() for item in self.intrinsic_depth.split('\n')[:-1]]
            self.intrinsic_depth = np.matrix(self.intrinsic_depth, dtype='float')
            self.img_folder_path = osp.join(args.scannet_path, 'color')
        elif self.args.dataset == 'visual_genome':
            self.img_folder_path = 'data/visual_genome/images'
            self.intrinsic_depth = None
        elif self.args.dataset in self.mot_benchmark_train:
            mot_train_path = osp.join(args.mot_benchmark_path, 'train')
            self.img_folder_path = osp.join(mot_train_path,self.args.dataset,'img1')
            self.intrinsic_depth = None
        elif self.args.dataset in self.mot_benchmark_test:
            mot_test_path = osp.join(args.mot_benchmark_path, 'test')
            self.img_folder_path = osp.join(mot_test_path, self.args.dataset,'img1')
            self.intrinsic_depth = None
        else:
            raise NotImplementedError

        self.num_frames = len(os.listdir(self.img_folder_path))

    def load_image(self,frame_idx):

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
            camera_pose = np.array(camera_pose,dtype=float)
            p_matrix = [ self.intrinsic_color[0][:], self.intrinsic_color[1][:], self.intrinsic_color[2][:]]
            p_matrix = np.matrix(p_matrix, dtype='float')
            inv_p_matrix = np.linalg.pinv(p_matrix)
            R = np.matrix([camera_pose[0][0:3], camera_pose[1][0:3], camera_pose[2][0:3]], dtype='float')
            inv_R = np.linalg.inv(R)
            Trans = np.matrix([camera_pose[0][3], camera_pose[1][3], camera_pose[2][3]], dtype='float')
            img_scene = cv2.imread(img_path)
            return img_scene, depth_img, pix_depth, inv_p_matrix, inv_R, Trans, camera_pose

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


def set_options(args):
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

    return options
