from __future__ import division
import numpy as np
import os
import cv2
import math
import time
import matplotlib.pyplot as plt
import random
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



def blurryness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def overlap(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

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
        curr_blurry = blurryness(cv2.imread(file_name_img[ind]))
        average_of_blurryness = alpha * average_of_blurryness + (1 - alpha) * curr_blurry
        threshold = 25 * math.log(average_of_blurryness) + 25
        print('Blurry Score (the higher the better): {score:3.1f} , Threshold: {threshold:3.1f}'
              .format(score=curr_blurry,threshold=threshold))
        #if curr_blurry < threshold:
        #    continue
        ckpt_blurry = time.time()

        # 2. calculate the relative pose to key & anchor frames
        curr_pose = read_matrix_from_txt(pose_path)
        rel_pose_to_key = relative_pose(curr_pose, key_pose)
        rel_pose_to_anchor = relative_pose(curr_pose, anchor_pose)
        ckpt_calc_pose = time.time()


        # 3. calculate the ratio of the overlapping area
        depth = cv2.imread(depth_path, -1)
        overlap_with_key,coordinates = calculate_overlap(depth, rel_pose_to_key, intrinsic, pixel_coordinates)
        overlap_with_anchor,coordinates = calculate_overlap(depth, rel_pose_to_anchor, intrinsic, pixel_coordinates)
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
        #print('--------------Elapsed Time---------------')
        print('blurry  pose   overlap  update')
        print('{blurry:1.2f}    {pose:1.2f}    {overlap:1.2f}    {update:1.2f} \n'.format(
            blurry=ckpt_blurry-time_start, pose=ckpt_calc_pose-ckpt_blurry,
            overlap = ckpt_calc_overlap - ckpt_calc_pose, update = ckpt_update-ckpt_calc_overlap))

    return key_frame_groups


def warp_image(target, depth, rel_pose, intrinsic):
    height, width, _  = target.shape
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


def calculate_overlap(depth, pose, intrinsic,pixel_coordinates):
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
    start_overlap = time.time()
    height, width = depth.shape
    #x, y = np.arange(width), np.arange(height)
    #X, Y = np.meshgrid(x, y)
    #pixel_coordinates = [list(zip(x, y)) for x, y in zip(X, Y)]
    #pixel_coordinates = [np.array([x, y, 1]).reshape(3, 1) for sub in pixel_coordinates for x, y in sub]
    #pixel_coordinates = np.array([[x, y, 1] for x in np.arange(height) for y in np.arange(width)])
    # pixel_coordinates = np.array(pixel_coordinates).squeeze()
    #pixel_coordinates = np.swapaxes(pixel_coordinates, 0, 1)
    ckpt_step1 = time.time()


    ## Step 2. pixel coordinate => camera coordinate (real-world coordinate)
    intrinsic_inv = np.linalg.inv(intrinsic)
    coordinates = np.dot(intrinsic_inv, pixel_coordinates)
    coordinates = np.swapaxes(coordinates, 0, 1)
    # coordinates = [np.dot(intrinsic_inv, pixel_coord) for pixel_coord in pixel_coordinates]
    # coordinates = np.array(coordinates).squeeze()
    ckpt_step2 = time.time()


    ## Step 3. 3-D position reconstruction
    depth = depth.flatten().reshape(-1, 1)
    coordinates = np.multiply(coordinates, depth)
    ckpt_step3 = time.time()



    ## Step 4. Reprojection
    # homogeneous coordinate for 3-D points
    coordinates = np.hstack((coordinates, np.ones((coordinates.shape[0], 1))))
    coordinates = np.swapaxes(coordinates, 0, 1)
    coordinates = np.swapaxes(np.dot(pose, coordinates), 0, 1)
    # normalization for 3-D points
    coordinates = coordinates[:, :3] / (coordinates[:, 3][:, None] + 1e-10)
    # reprojection
    coordinates = np.swapaxes(coordinates, 0, 1)
    coordinates = np.swapaxes(np.dot(intrinsic, coordinates), 0, 1)
    # normalization for 2-D points
    coordinates = coordinates[:, :2] / (coordinates[:, 2][:, None] + 1e-10)
    ckpt_step4 = time.time()


    ## Step 5. Calculate the amount of the overlapping area
    ov2_start = time.time()
    overlapping_area2 = sum((coordinates[:, 0] < width) & (coordinates[:, 1] < height)
                            & (coordinates[:, 0] > 0) & (coordinates[:, 1] > 0))
    overlapping_area2 /= (width * height)
    ov2_end = time.time()


    ov1_start = time.time()
    # Randomly sample 1000 points
    coordinates = random.sample(coordinates, 1000)
    coordinates = np.stack(coordinates)

    overlapping_points = coordinates[(coordinates[:, 0] < width) & (coordinates[:, 1] < height)\
                           & (coordinates[:, 0] > 0) & (coordinates[:, 1] > 0)]
    minX, minY = overlapping_points.min(axis=0)
    maxX, maxY = overlapping_points.max(axis=0)

    overlapping_area1 = (maxX-minX)*(maxY-minY)/(width*height)
    overlapping_area1 = min(overlapping_area1, 1.0)
    ov1_end = time.time()




    ckpt_step5 = time.time()

    #mean,std = coordinates.mean(axis=0), coordinates.std(axis=0)
    #coordinates = coordinates[coordinates[:,0] < mean[0] + 1 * std[0]]
    #coordinates = coordinates[coordinates[:,1] > mean[1] - 1 * std[1]]

    # p1 = coordinates[np.argmax(coordinates[:, 0])]
    # p2 = coordinates[np.argmax(coordinates[:, 1])]
    # p3 = coordinates[np.argmin(coordinates[:, 0])]
    # p4 = coordinates[np.argmin(coordinates[:, 1])]
    #
    # quadrangle = np.stack([p1,p2,p3,p4])

    print('overlap1: {ov1:1.2f}  overlap2: {ov2:1.2f}  '.format(ov1=overlapping_area1, ov2= overlapping_area2))

    print('--------------Elapsed Time---------------')
    print('method1:  {ov1:1.2f}  method2:  {ov2:1.2f}  '.format(ov1=ov1_end-ov1_start, ov2=ov2_end-ov2_start))

    print('--------------Elapsed Time---------------')
    print('step1    step2    step3    step4    step5')
    print('{s1:1.3f}    {s2:1.3f}    {s3:1.3f}    {s4:1.3f}    {s5:1.3f} \n'.format(
        s1=ckpt_step1 - start_overlap, s2=ckpt_step2 - ckpt_step1,
        s3=ckpt_step3 - ckpt_step2, s4=ckpt_step4 - ckpt_step3,s5=ckpt_step5 - ckpt_step4))


    # plt.clf()
    # plt.scatter(coordinates[:, 0].tolist(), coordinates[:, 1].tolist(),s=1,color='b')
    # plt.scatter(quadrangle[:,0].tolist(),quadrangle[:,1].tolist(),s=10,color='r')
    # #plt.xlim([-3000,3000])
    # #plt.ylim([-3000,3000])
    # #plt.show()
    # plt.draw()
    # plt.pause(0.0001)


    return overlapping_area2, coordinates


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


file_name_img, file_name_depth, file_name_pose = read_files(PATH_IMG), read_files(PATH_DEPTH), read_files(PATH_POSE)
#
# ref_idx = 10
intrinsic = read_matrix_from_txt(PATH_INTRINSIC)[:3, :3] # 3x3 matrix
# pose2 = read_matrix_from_txt(file_name_pose[ref_idx]) # 4x4 matrix
# img_test = cv2.imread(file_name_img[ref_idx]) # 480x640x3 RGB image
# test_depth= cv2.imread(file_name_depth[ref_idx], -1) # 480x640 Depth image
# hi, wi, _ = img_test.shape
# hd, wd = test_depth.shape
#
# x_ratio = wd / wi
# y_ratio = hd / hi
#
# intrinsic[0] *= x_ratio
# intrinsic[1] *= y_ratio
#
# img_test = cv2.resize(img_test, (wd, hd))
#
# test_idx = 300
# img_test2 = cv2.imread(file_name_img[test_idx])
# depth, pose = cv2.imread(file_name_depth[test_idx], -1), read_matrix_from_txt(file_name_pose[test_idx])
# rel_pose = relative_pose(pose2, pose) # pose * rel_pose = pose2
# test = calculate_overlap(depth, rel_pose, intrinsic)
# rel_pose2 = relative_pose(pose, pose2)
#
# img_test2 = cv2.resize(img_test2, (wd, hd))
#
# weird_pose = read_matrix_from_txt(file_name_pose[1000])
# rel_pose3 = relative_pose(pose, weird_pose)

# out_img1 = warp_image(img_test2, test_depth, rel_pose, intrinsic)
# out_img2 = warp_image(img_test2, test_depth, rel_pose2, intrinsic)

# cv2.imshow('target_image', img_test2)
# cv2.imshow('source_image', img_test)
# cv2.imshow('warped_image1 with rel_pose', out_img1)
# cv2.imshow('warped_image1 with rel_pose2', out_img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
for test_idx in range(1, 5570, 10):
    depth, pose = cv2.imread(file_name_depth[test_idx], -1), read_matrix_from_txt(file_name_pose[test_idx])
    rel_pose = relative_pose(pose, pose2) # pose * rel_pose = pose2
    test = calculate_overlap(depth, rel_pose, intrinsic)
    print('test_idx: {}, overlap: {}'.format(test_idx, test))
'''

plt.axis([-3000, 3000, -3000, 3000])
plt.ion()
plt.show()
keyframe_groups = key_frame_extractor(file_name_img, file_name_depth, file_name_pose, intrinsic)
