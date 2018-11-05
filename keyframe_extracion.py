from __future__ import division
import numpy as np
import os
import os.path as osp
import cv2
import math
import time
import matplotlib.pyplot as plt
import random

PATH_IMG = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/color/'
PATH_DEPTH = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/depth/'
PATH_POSE = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/pose/'
PATH_INTRINSIC = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/intrinsic/intrinsic_depth.txt'


def read_files(path):
    """
        Description
            - Read files in the given directory and return the list of the paths of the files

        Parameter
            - path: path to the directory to be read

        Return
            - file_name: list of file paths

    """
    file_name = os.listdir(path)
    file_name = sorted(file_name, key=lambda x: int(x.split('.')[0]))
    file_name = [path + fn for fn in file_name]
    return file_name


def blurryness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def warp_image(target, depth, rel_pose, intrinsic):
    height, width, _ = target.shape
    output = np.zeros((height, width, 3), np.uint8)

    for i in range(height):
        for j in range(width):
            temp = np.dot(np.linalg.inv(intrinsic), np.array([i, j, 1], dtype=float).reshape(3, 1))
            temp = (depth[i, j]) * temp
            temp = np.dot(rel_pose, np.append(temp, [1]).reshape(4, 1))
            temp = temp / temp[3]
            temp = np.dot(intrinsic, temp[:3])
            temp = temp / temp[2]
            x, y = int(round(temp[0])), int(round(temp[1]))
            if x >= 0 and x < height and y >= 0 and y < width:
                output[i, j, :] = target[x, y, :]
    return output


def calculate_overlap(depth, pose, intrinsic, pixel_coordinates,num_coordinate_samples=1000):
    """
        Description
            - Calculate overlap between two images based on projection
            - Projection of img2 to img1
                - p' = K * T_21 * depth * K^(-1) * p

        Parameter
            - depth: information on depth image
            - pose: relative pose to the reference image
            - intrinsic: camera intrinsic

        Return
            - amount_overlap: estimated amount of overlap in percentage

    """

    # print('depth')
    # print(depth, depth.shape, depth.dtype)
    # print('pose')
    # print(pose, pose.shape, pose.dtype)
    # print('intrinsic')
    # print(intrinsic, intrinsic.shape, intrinsic.dtype)


    ## Step 1. Pixel coordinates (p in the above eq.)
    height, width = depth.shape

    ## Step 2. pixel coordinate => camera coordinate (real-world coordinate)
    intrinsic_inv = np.linalg.inv(intrinsic)
    coordinates = np.dot(intrinsic_inv, pixel_coordinates)
    coordinates = np.swapaxes(coordinates, 0, 1)

    ## Step 3. 3-D position reconstruction
    depth = depth.flatten().reshape(-1, 1)
    coordinates = np.multiply(coordinates, depth)

    ## Step 4. Reprojection
    # homogeneous coordinate for 3-D points
    coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))
    coordinates = np.swapaxes(coordinates, 0, 1)
    coordinates = np.swapaxes(np.dot(pose, coordinates), 0, 1)
    # normalization for 3-D points
    # print(coordinates, coordinates.shape, coordinates.dtype)
    coordinates = coordinates[:, :3] / (coordinates[:, 3][:, None] + 1e-10)
    # reprojection
    coordinates = np.swapaxes(coordinates, 0, 1)
    coordinates = np.swapaxes(np.dot(intrinsic, coordinates), 0, 1)
    # normalization for 2-D points
    coordinates = coordinates[:, :2] / (coordinates[:, 2][:, None] + 1e-10)


    ## Step 5. Calculate the amount of the overlapping area
    # Randomly sample 1000 points
    coordinates = random.sample(coordinates, num_coordinate_samples)
    coordinates = np.stack(coordinates)
    overlapping_points = coordinates[(coordinates[:, 0] < width) & (coordinates[:, 1] < height) \
                                     & (coordinates[:, 0] > 0) & (coordinates[:, 1] > 0)]
    if overlapping_points.size > 0:
        minX, minY = overlapping_points.min(axis=0)
        maxX, maxY = overlapping_points.max(axis=0)

        overlapping_area = (maxX - minX) * (maxY - minY) / (width * height)
        overlapping_area = min(overlapping_area, 1.0)
    else:
        overlapping_area = 0.0

    return overlapping_area


def relative_pose(pose1, pose2):
    """
        Description
            - Calculate relative pose between a pair of poses
            - To avoid calculating matrix inverse, the calculation is based on
                - P_12 = [R_2^(-1) R_2^(-1)(t_1 - t_2); 0, 0, 0, 1],
                - where R_2^(-1) = R_2.T

        Parameter
            - pose1, pose2: 4 x 4 pose matrix

        Return
            - p_2_to_1 (relative_pose): estimated relative pose

    """
    """
    R_1, R_2 = pose1[:3, :3], pose2[:3, :3]
    t_1, t_2 = pose1[:, -1][:-1], pose2[:, -1][:-1]
    R = np.dot(R_2.T, R_1)
    T = np.dot(R_2.T, t_1 - t_2)
    p_1_to_2 = np.zeros((4, 4))
    p_1_to_2[:3, :3] = R
    p_1_to_2[:3, -1] = T
    p_1_to_2[-1, -1] = 1
    """
    p_2_to_1 = np.dot(np.linalg.inv(pose2), pose1)
    return p_2_to_1


def read_matrix_from_txt(matrix_file):
    """
        Description
            - Read a matrix from .txt file

        Parameter
            - matrix_file: .txt file containing n x m matrix

        Return
            - matrix_array: numpy array of (n, m) shape

    """
    f = open(matrix_file).readlines()
    matrix_array = [row.split() for row in f]
    matrix_array = np.array(matrix_array, dtype=float)
    return matrix_array


def key_frame_extractor(file_name_img, file_name_depth, file_name_pose, intrinsic):
    """
        Description
            - Extract keyframe groups by calculating overlapping areas

        Parameter
            - file_name_depth: list of depth paths
            - file_name_pose: list of pose paths
            - cam_intrinsic: scaled camera intrinsic matrix

        Return
            - key_frame_groups: extracted key frame groups

    """
    # assert len(file_name_img) == len(file_name_depth), "Number of image != number of depth"
    # assert len(file_name_img) == len(file_name_pose), "Number of image != number of pose"
    # assert len(file_name_depth) == len(file_name_pose), "Number of depth != number of pose"

    # initialize variables
    key_frame_groups, curr_key_frame_group = [], [0]
    key_pose, anchor_pose = [read_matrix_from_txt(file_name_pose[0])] * 2
    thresh_key, thresh_anchor = 0.1, 0.65
    average_of_blurryness = blurryness(cv2.imread(file_name_img[0]))
    alpha = 0.7
    height, width = 480, 640
    pixel_coordinates = np.array([[x, y, 1] for x in np.arange(height) for y in np.arange(width)])
    pixel_coordinates = np.swapaxes(pixel_coordinates, 0, 1)

    for ind, (pose_path, depth_path) in enumerate(zip(file_name_pose, file_name_depth)):
        print('FrameNum: {ind:4}'.format(ind=ind))
        # 1. reject blurry images
        time_start = time.time()
        rgb_image =cv2.imread(file_name_img[ind])
        depth = cv2.imread(depth_path, -1)
        # hi, wi, _ = rgb_image.shape
        hd, wd = depth.shape
        #
        # x_ratio = wd / wi
        # y_ratio = hd / hi
        #
        # intrinsic[0] *= x_ratio
        # intrinsic[1] *= y_ratio
        #
        #rgb_image = cv2.resize(rgb_image, (wd,hd), interpolation= cv2.INTER_AREA)
        #cv2.imshow('sample',rgb_image)
        #cv2.waitKey(1)

        curr_blurry = blurryness(rgb_image)
        average_of_blurryness = alpha * average_of_blurryness + (1 - alpha) * curr_blurry
        threshold = 25 * math.log(average_of_blurryness) + 25
        print('Blurry Score (the higher the better): {score:3.1f} , Threshold: {threshold:3.1f}'
              .format(score=curr_blurry, threshold=threshold))
        if curr_blurry < threshold:
           continue
        ckpt_blurry = time.time()

        # 2. calculate the relative pose to key & anchor frames
        curr_pose = read_matrix_from_txt(pose_path)
        rel_pose_to_key = relative_pose(curr_pose, key_pose)
        rel_pose_to_anchor = relative_pose(curr_pose, anchor_pose)
        ckpt_calc_pose = time.time()

        # 3. calculate the ratio of the overlapping area

        overlap_with_key = calculate_overlap(depth, rel_pose_to_key, intrinsic, pixel_coordinates)
        overlap_with_anchor = calculate_overlap(depth, rel_pose_to_anchor, intrinsic, pixel_coordinates)
        ckpt_calc_overlap = time.time()

        # 4. update anchor and key frames
        if overlap_with_anchor < thresh_anchor:
            curr_key_frame_group.append(ind)
            anchor_pose = curr_pose

        if overlap_with_key < thresh_key or len(curr_key_frame_group) > 10:
            key_frame_groups.append(curr_key_frame_group)
            curr_key_frame_group = []
            key_pose, anchor_pose = [curr_pose] * 2

        ckpt_update = time.time()

        print('overlap_with_key: {ov1:1.2f}  overlap_with_anchor: {ov2:1.2f}  '
              .format(ov1=overlap_with_key, ov2=overlap_with_anchor))

        # print('--------------Elapsed Time---------------')
        print('blurry  pose   overlap  update')
        print('{blurry:1.2f}    {pose:1.2f}    {overlap:1.2f}    {update:1.2f} \n'.format(
            blurry=ckpt_blurry - time_start, pose=ckpt_calc_pose - ckpt_blurry,
            overlap=ckpt_calc_overlap - ckpt_calc_pose, update=ckpt_update - ckpt_calc_overlap))

        # cv2.namedWindow('sample')  # Create a named window
        # cv2.moveWindow('sample', 700, 10)
        # cv2.imshow('sample', rgb_image)
        # cv2.waitKey(1)



    return key_frame_groups


class keyframe_checker(object):
    def __init__(self,args,
                 intrinsic_depth = None,
                 thresh_key=0.1,
                 thresh_anchor=0.65,
                 max_group_len = 10,
                 blurry_gain=30,
                 blurry_offset=10,
                 alpha=0.4, depth_shape=(480, 640) ,
                 num_coordinate_samples=1000,
                 BLURRY_REJECTION_ONLY = None):
        self.args = args
        self.frame_num = 0
        if BLURRY_REJECTION_ONLY == None:
            self.BLURRY_REJECTION_ONLY = True if self.args.dataset != 'scannet' else False
        else: self.BLURRY_REJECTION_ONLY=BLURRY_REJECTION_ONLY

        # Blurry Image Rejection: Hyper parameters
        self.blurry_gain = blurry_gain
        self.blurry_offset = blurry_offset
        self.alpha = alpha

        # Key frame/ anchor frame Selection: Hyper parameters
        if not self.BLURRY_REJECTION_ONLY:
            self.intrinsic_depth = np.array(intrinsic_depth[:3,:3])
            self.thresh_key = thresh_key
            self.thresh_anchor = thresh_anchor
            self.max_group_len = max_group_len
            self.depth_shape = depth_shape
            pixel_coordinates = np.array([[x, y, 1] for x in np.arange(depth_shape[0]) for y in np.arange(depth_shape[1])])
            self.pixel_coordinates = np.swapaxes(pixel_coordinates, 0, 1)
            self.key_frame_groups, self.curr_key_frame_group = [], [0]
            self.num_cooridnate_samples = num_coordinate_samples



    def check_frame(self,img, depth, pose):
        if self.args.disable_keyframe: return True, 0.0, 0.0
        if self.frame_num == 0:
            self.average_of_blurryness = blurryness(img)
            self.key_pose, self.anchor_pose, = [pose] * 2

        # 1. reject blurry images
        curr_blurry = blurryness(img)
        self.average_of_blurryness = self.alpha * self.average_of_blurryness + (1 - self.alpha) * curr_blurry
        threshold = self.blurry_gain * math.log(self.average_of_blurryness) + self.blurry_offset
        #threshold = self.blurry_gain * self.average_of_blurryness + self.blurry_offset
        if curr_blurry < threshold: return False, curr_blurry, threshold
        #if self.BLURRY_REJECTION_ONLY: return curr_blurry > threshold, curr_blurry, threshold

        # 2. calculate the relative pose to key & anchor frames
        rel_pose_to_key = relative_pose(pose, self.key_pose)
        rel_pose_to_anchor = relative_pose(pose, self.anchor_pose)

        # 3. calculate the ratio of the overlapping area
        depth = np.asarray(depth,dtype='uint16')

        overlap_with_key = calculate_overlap(depth, rel_pose_to_key, self.intrinsic_depth,
                                             self.pixel_coordinates,self.num_cooridnate_samples)
        overlap_with_anchor = calculate_overlap(depth, rel_pose_to_anchor, self.intrinsic_depth,
                                                self.pixel_coordinates,self.num_cooridnate_samples)

        # 4. update anchor and key frames
        if overlap_with_anchor < self.thresh_anchor:
            self.curr_key_frame_group.append(self.frame_num)
            self.anchor_pose = pose
            IS_ANCHOR = True
        else: IS_ANCHOR = False

        if overlap_with_key < self.thresh_key or len(self.curr_key_frame_group) > self.max_group_len:
            self.key_frame_groups.append(self.curr_key_frame_group)
            self.curr_key_frame_group = []
            self.key_pose, self.anchor_pose = [pose] * 2
            IS_KEY = True
        else: IS_KEY = False

        self.frame_num +=1

        return IS_KEY or IS_ANCHOR, curr_blurry, threshold




if __name__=="__main__":
    file_name_img, file_name_depth, file_name_pose = read_files(PATH_IMG), read_files(PATH_DEPTH), read_files(PATH_POSE)
    intrinsic = read_matrix_from_txt(PATH_INTRINSIC)[:3, :3]  # 3x3 matrix
    keyframe_groups = key_frame_extractor(file_name_img[:1000], file_name_depth[:1000], file_name_pose[:1000], intrinsic)
