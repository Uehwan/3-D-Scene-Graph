from __future__ import division
import numpy as np
import os
import cv2


PATH_IMG = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/color/'
PATH_DEPTH = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/depth/'
PATH_POSE = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/pose/'
PATH_INTRINSIC = '/media/mil2/HDD/mil2/scannet/ScanNet/SensReader/python/exported/intrinsic/intrinsic_color.txt'


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


file_name_img, file_name_depth, file_name_pose = read_files(PATH_IMG), read_files(PATH_DEPTH), read_files(PATH_POSE)


def key_frame_extractor(file_name_depth, file_name_pose, cam_intrinsic):
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
    assert len(file_name_img) == len(file_name_depth), "Number of image != number of depth"
    assert len(file_name_img) == len(file_name_pose), "Number of image != number of pose"
    assert len(file_name_depth) == len(file_name_pose), "Number of depth != number of pose"

    # initialize variables
    key_frame_groups, curr_key_frame_group = [], [0]
    key_pose, anchor_pose = [read_matrix_from_txt(file_name_pose[0])] * 2
    thresh_key, thresh_anchor = 0.2, 0.8

    for ind, (pose, depth) in enumerate(zip(file_name_pose, file_name_depth)):
        # calculate the relative pose to key & anchor frames
        curr_pose = read_matrix_from_txt(pose)
        rel_pose_to_key = relative_pose(key_pose, curr_pose)
        rel_pose_to_anchor = relative_pose(anchor_pose, curr_pose)

        # calculate the ratio of the overlapping area
        depth = cv2.imread(depth, -1)
        overlap_with_key = calculate_overlap(depth, rel_pose_to_key, cam_intrinsic)
        overlap_with_anchor = calculate_overlap(depth, rel_pose_to_anchor, cam_intrinsic)
        
        # update anchor and key frames
        if overlap_with_anchor < thresh_anchor:
            curr_key_frame_group.append(ind)
            anchor_pose = curr_pose

        if overlap_with_key < thresh_key:
            key_frame_groups.append(curr_key_frame_group)
            curr_key_frame_group = []
            key_pose, anchor_pose = [curr_pose] * 2
    return key_frame_groups


def calculate_overlap(depth, pose, intrinsic):
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
    ## Step 1. Pixel coordinates (p in the above eq.)
    height, width = depth.shape
    #x, y = np.arange(width), np.arange(height)
    #X, Y = np.meshgrid(x, y)
    #pixel_coordinates = [list(zip(x, y)) for x, y in zip(X, Y)]
    #pixel_coordinates = [np.array([x, y, 1]).reshape(3, 1) for sub in pixel_coordinates for x, y in sub]
    pixel_coordinates = np.array([[x, y, 1] for x in np.arange(height) for y in np.arange(width)])
    # pixel_coordinates = np.array(pixel_coordinates).squeeze()
    pixel_coordinates = np.swapaxes(pixel_coordinates, 0, 1)

    ## Step 2. pixel coordinate => camera coordinate (real-world coordinate)
    intrinsic_inv = np.linalg.inv(intrinsic)
    coordinates = np.dot(intrinsic_inv, pixel_coordinates)
    coordinates = np.swapaxes(coordinates, 0, 1)
    # coordinates = [np.dot(intrinsic_inv, pixel_coord) for pixel_coord in pixel_coordinates]
    # coordinates = np.array(coordinates).squeeze()

    ## Step 3. 3-D position reconstruction
    depth = depth.flatten().reshape(-1, 1)
    coordinates = np.multiply(coordinates, depth)
    
    ## Step 4. Reprojection
    # homogeneous coordinate for 3-D points
    coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))
    coordinates = np.swapaxes(coordinates, 0, 1)
    coordinates = np.swapaxes(np.dot(pose, coordinates), 0, 1)
    # normalization for 3-D points
    coordinates = coordinates[:, :3] / coordinates[:, 3][:, None]
    # reprojection
    coordinates = np.swapaxes(coordinates, 0, 1)
    coordinates = np.swapaxes(np.dot(intrinsic, coordinates), 0, 1)
    # normalization for 2-D points
    coordinates = coordinates[:, :2] / coordinates[:, 2][:, None]

    ## Step 5. Calculate the amount of the overlapping area
    depth_zero = (depth == 0)
    overlapping_area = sum((coordinates[:, 0] < width) & (coordinates[:, 1] < height)
                           & (coordinates[:, 0] > 0) & (coordinates[:, 1] > 0))
    overlapping_area /= (width * height)
    overlapping_area = min(overlapping_area, 1.0)
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


ref_idx = 1
intrinsic = read_matrix_from_txt(PATH_INTRINSIC)[:3, :3]
pose2 = read_matrix_from_txt(file_name_pose[ref_idx])
img_test = cv2.imread(file_name_img[ref_idx])
test_depth= cv2.imread(file_name_depth[10], -1)
hi, wi, _ = img_test.shape
hd, wd = test_depth.shape

x_ratio = wd / wi
y_ratio = hd / hi

intrinsic[0] *= x_ratio
intrinsic[1] *= y_ratio

'''
test_idx = 300
depth, pose = cv2.imread(file_name_depth[test_idx], -1), read_matrix_from_txt(file_name_pose[test_idx])
rel_pose = relative_pose(pose2, pose) # pose * rel_pose = pose2
test = calculate_overlap(depth, rel_pose, intrinsic)

'''

for test_idx in range(1, 5570, 10):
    depth, pose = cv2.imread(file_name_depth[test_idx], -1), read_matrix_from_txt(file_name_pose[test_idx])
    rel_pose = relative_pose(pose, pose2) # pose * rel_pose = pose2
    test = calculate_overlap(depth, rel_pose, intrinsic)
    print('test_idx: {}, overlap: {}'.format(test_idx, test))


