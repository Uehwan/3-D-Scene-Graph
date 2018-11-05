import cv2
import random
import numpy as np
from pandas import DataFrame
import pandas as pd
from graphviz import Digraph
import webcolors
import pprint
import math
from scipy.stats import norm
from color_histogram.core.hist_3d import Hist3D
#import pcl # cd python-pcl -> python setup.py build-ext -i -> python setup.py install
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torchtext #0. install torchtext==0.2.3 (pip install torchtext==0.2.3)
from torch.nn.functional import cosine_similarity
from collections import Counter
import pcl
import os.path as osp
import os
fasttext = torchtext.vocab.FastText()
_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)



def vis_bbox_opencv(img, bbox, color=_GREEN,thick=1):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness=thick)
    return img

def get_class_string(class_index, score, dataset):
    class_text = dataset[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def compare_class(curr_cls, prev_cls, cls_score,compare_all=False):
    similar_cls = False
    same_cls = 0
    score = 0
    for cls in curr_cls:
        if cls in prev_cls:
            similar_cls = True
            same_cls += 1

    if (similar_cls):
        #score = float(same_cls) * cls_score
        if (same_cls == 3):
            score = 1.0
        if (same_cls == 2):
            score = 0.9
        if (same_cls == 1):
            score = 0.8

    # max similarity = 1 (same word), min similarity = 0 (no relation)
    # similarity = cosine_similarity(fasttext.vectors[fasttext.stoi['hello']],fasttext.vectors[fasttext.stoi['hi']],dim=0)
    else:
        if compare_all:
            similarity = 0
            for i in range(3):
                for j in range(3):
                    similarity += cosine_similarity(fasttext.vectors[fasttext.stoi[curr_cls[i]]].cuda(),fasttext.vectors[fasttext.stoi[prev_cls[j]]].cuda(),dim=0).cpu()[0]
            similarity /= 9.0
        #print(similarity)
        else:
            similarity = cosine_similarity(fasttext.vectors[fasttext.stoi[curr_cls[0]]].cuda(),
                                        fasttext.vectors[fasttext.stoi[prev_cls[0]]].cuda(), dim=0).cpu()[0]

        #score = similarity * cls_score
        score = 0.

    return score

def compare_position(curr_mean, curr_var, prev_mean, prev_var, prev_pt_num, new_pt_num):
    I_x, I_y, I_z = check_distance(curr_mean,curr_var, prev_mean, prev_var) 
    #score = (I_x * I_y * I_z)
    score = (I_x/3.0) + (I_y/3.0) + (I_z/3.0)
    score = float(score)
    return score

def compare_color(curr_hist, prev_hist):
    curr_rgb = webcolors.name_to_rgb(curr_hist[0][1])
    prev_rgb = webcolors.name_to_rgb(prev_hist[0][1])
    dist = np.sqrt(np.sum(np.power(np.subtract(curr_rgb, prev_rgb),2))) / (255*np.sqrt(3))
    score = 1-dist
    return score
     
def check_distance(x,curr_var, mean, var):
    Z_x = (x[0]-mean[0])/1000.
    Z_y = (x[1]-mean[1])/1000.
    Z_z = (x[2]-mean[2])/1000.
    #Z_x = (x[0]-mean[0])/np.sqrt(curr_var[0])
    #Z_y = (x[1]-mean[1])/np.sqrt(curr_var[1])
    #Z_z = (x[2]-mean[2])/np.sqrt(curr_var[2])
    # In Standardized normal gaussian distribution
    # Threshold : 0.9 --> -1.65 < Z < 1.65
    #           : 0.8 --> -1.29 < Z < 1.29
    #           : 0.7 --> -1.04 < Z < 1.04
    #           : 0.6 --> -0.845 < Z < 0.845
    #           : 0.5 --> -0.675 < Z < 0.675
    #           : 0.4 --> -0.53 < Z < 0.53
    #print("    pos {pos_x:3.2f} {pose_y:3.2f} {pose_z:3.2f}".format(pos_x=Z_x, pose_y=Z_y, pose_z=Z_z))
    #print("pos_y {pos_y:3.2f}".format(pos_y=Z_y))
    #print("pos_z {pos_z:3.2f}".format(pos_z=Z_z))
    th_x = 0.9
    th_y = 0.9
    th_z = 0.9
    #th_x = np.sqrt(np.abs(var[0])) *beta
    #th_y = np.sqrt(np.abs(var[1])) *beta
    #th_z = np.sqrt(np.abs(var[2])) *beta
    x_check = -th_x < Z_x < th_x
    y_check = -th_y < Z_y < th_y
    z_check = -th_z < Z_z < th_z
    
    if (x_check):
        I_x = 1.0
    else:
        #I_x = norm.cdf(-np.abs(Z_x)) / norm.cdf(-th_x)
        #I_x = (norm.cdf(th_x) - norm.cdf(-th_x)) / (norm.cdf(np.abs(Z_x)) - norm.cdf(-np.abs(Z_x)))
        I_x = th_x / np.abs(Z_x)
        # if (np.abs(th_x - Z_x)<1):
        #     I_x = np.abs(th_x - Z_x)
        # else:
        #     I_x = 1/np.abs(th_x-Z_x)
    if (y_check):
        I_y = 1.0
    else:
        #I_y = norm.cdf(-np.abs(Z_y)) / norm.cdf(-th_y)
        #I_y = (norm.cdf(th_y) - norm.cdf(-th_y)) / (norm.cdf(np.abs(Z_y)) - norm.cdf(-np.abs(Z_y)))
        I_y = th_y / np.abs(Z_y)
        # if (np.abs(th_y - Z_y)<1):
        #     I_y = np.abs(th_y - Z_y)
        # else:
        #     I_y = 1/np.abs(th_y-Z_y)
    if (z_check):
        I_z = 1.0
    else:
        #I_z = norm.cdf(-np.abs(Z_z)) / norm.cdf(-th_z)
        #I_z = (norm.cdf(th_z) - norm.cdf(-th_z)) / (norm.cdf(np.abs(Z_z)) - norm.cdf(-np.abs(Z_z)))
        I_z = th_z / np.abs(Z_z)
        # if (np.abs(th_z - Z_z)<1):
        #     I_z = np.abs(th_z - Z_z)
        # else:
        #     I_z = 1/np.abs(th_x-Z_z)
    
    #print("    score {score_x:3.2f} {score_y:3.2f} {score_z:3.2f}".format(score_x=I_x, score_y=I_y, score_z=I_z))
    #print("    tot_score {score:3.2f} ".format(score=(I_x+I_y+I_z)/3.))
    #print("pose_score_y {pos_score_y:3.2f}".format(pos_score_y=I_y))
    #print("pose_score_z {pos_score_z:3.2f}".format(pos_score_z=I_z))
    return I_x, I_y, I_z


def node_update(window_3d_pts, global_node, curr_mean, curr_var,  curr_cls, cls_score, curr_color_hist,test_set ):
    try:
        new_pt_num = len(window_3d_pts)
        global_node_num = len(global_node)
        print(global_node_num)
        score = []
        score_pose = []
        cls_score = cls_score[0]
        w1, w2, w3 = 10.0/20.0, 8.0/20.0, 2.0/20.0
        #print("current object : {cls:3}".format(cls=curr_cls[0]))
        for i in range(global_node_num):
            prev_cls = (-global_node.ix[i]["class"]).argsort()[:3] # choose top 3 index
            prev_cls =  [test_set.object_classes[ind] for ind in prev_cls] # index to text
            #print("compare object : {comp_cls:3}".format(comp_cls=prev_cls[0]))
            prev_mean, prev_var, prev_pt_num = global_node.ix[i]["mean"], global_node.ix[i]["var"], global_node.ix[i]["pt_num"]
            prev_color_hist = global_node.ix[i]["color_hist"]
            cls_sc = compare_class(curr_cls, prev_cls, cls_score)
            pos_sc = compare_position(curr_mean,curr_var, prev_mean, prev_var, prev_pt_num, new_pt_num)
            col_sc = compare_color(curr_color_hist, prev_color_hist)
            #print("class_score {cls_score:3.2f}".format(cls_score=cls_sc))
            #print("pose_score {pos_score:3.2f}".format(pos_score=pos_sc))
            #print("color_score {col_score:3.2f}".format(col_score=col_sc))
            tot_sc = (w1 * cls_sc) + (w2 * pos_sc) + (w3 * col_sc)
            #print("total_score {tot_score:3.2f}".format(tot_score=tot_sc))
            score.append(tot_sc)
            #score_pose.append(pos_sc)
        node_score = max(score)
        print("node_score {score:3.4f}".format(score=node_score))
        max_score_index = score.index(max(score))
        #node_score_pose = score_pose[max_score_index]
        #print("node_score_pose {score_pose:3.2f}".format(score_pose=node_score_pose))
        return node_score, max_score_index
    except:
        return 0,0

def Measure_new_Gaussian_distribution(new_pts):
    try:
        pt_num = len(new_pts)
        mu = np.sum(new_pts, axis=0)/pt_num
        mean = [int(mu[0]), int(mu[1]), int(mu[2])]
        var = np.sum(np.power(new_pts, 2), axis=0)/pt_num - np.power(mu,2)
        var = [int(var[0]), int(var[1]), int(var[2])]
        return pt_num, mean, var
    except:
        return 1, [0,0,0], [1,1,1]




def Measure_added_Gaussian_distribution(new_pts, prev_mean, prev_var, prev_pt_num, new_pt_num):
    # update mean and variance
    pt_num = prev_pt_num + new_pt_num
    mu = np.sum(new_pts, axis=0)
    mean = np.divide((np.multiply(prev_mean,prev_pt_num) + mu),pt_num)
    mean = [int(mean[0]), int(mean[1]), int(mean[2])]
    var = np.subtract(np.divide((np.multiply((prev_var + np.power(prev_mean,2)),prev_pt_num) + np.sum(np.power(new_pts,2),axis=0)) ,pt_num), np.power(mean,2))
    var = [int(var[0]), int(var[1]), int(var[2])]
    return pt_num, mean, var

def Draw_connected_scene_graph(node_feature, relation, img_count, test_set, sg, idx,cnt_thres=2,view=True,save_path='./vis_result/'):
    # load all of saved node_feature
    # if struct ids are same, updated to newly typed object
    #print('node_feature:',node_feature)
    tile_idx = []
    handle_idx = []
    for node_num in range(len(node_feature)):
        if node_feature.ix[node_num]['detection_cnt'] < cnt_thres: continue
        if len(node_feature.ix[node_num]["color_hist"]) ==1:
            node_feature.at[node_num,"color_hist"].append(node_feature.ix[node_num]["color_hist"][0])
        box_color_bgr = colorlist[int(node_feature.ix[node_num]["idx"])]
        box_color_rgb = box_color_bgr[::-1]
        box_color_rgb = webcolors.name_to_rgb(node_feature.ix[node_num]["color_hist"][0][1])
        box_color_hex = webcolors.rgb_to_hex(box_color_rgb)
        obj_cls =  str(test_set.object_classes[node_feature.ix[node_num]["class"].argmax()])
        if ( obj_cls == "tile"):
            tile_idx.append(str(node_feature.ix[node_num]["idx"]))
        elif (obj_cls == "handle"):
            handle_idx.append(str(node_feature.ix[node_num]["idx"]))
        else:
            sg.attr('node', style="", color=box_color_hex)
            sg.node('struct'+str(node_feature.ix[node_num]["idx"]), '''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
              <TR>
                <TD PORT="'''+str(node_feature.ix[node_num]["idx"])+'''" ROWSPAN="2">'''+str(test_set.object_classes[node_feature.ix[node_num]["class"].argmax()])+'''</TD>
                <TD COLSPAN="1">ID</TD>
                <TD COLSPAN="2">color</TD>
                <TD COLSPAN="3">3D Position</TD>
                <TD PORT="'''+'-'+str(node_feature.ix[node_num]["idx"])+'''" ROWSPAN="2"> Rel </TD>
                
              </TR>
              <TR>
                <TD>'''+str(node_feature.ix[node_num]["idx"])+'''</TD>
                <TD>'''+str(node_feature.ix[node_num]["color_hist"][0][1])+'''</TD>
                <TD>'''+str(node_feature.ix[node_num]["color_hist"][1][1])+'''</TD>
                <TD>'''+str(node_feature.ix[node_num]["3d_pose"][0]) + '''</TD>
                <TD>'''+str(node_feature.ix[node_num]["3d_pose"][1])+'''</TD>
                <TD>'''+str(node_feature.ix[node_num]["3d_pose"][2])+'''</TD>
              </TR>
            </TABLE>>''')
    tile_idx = set(tile_idx)
    tile_idx = list(tile_idx)
    handle_idx = set(handle_idx)
    handle_idx = list(handle_idx)

    relation_list = []
    for num in range(len(relation)):    
        relation_list.append((relation.ix[num]["relation"][0], relation.ix[num]["relation"][1],relation.ix[num]["relation"][2]))

    relation_list = [rel for rel in relation_list if (node_feature.loc[node_feature['idx'] == int(rel[0])]['detection_cnt'].item()>=min(cnt_thres,idx))]
    relation_list = [rel for rel in relation_list if (node_feature.loc[node_feature['idx'] == int(rel[2])]['detection_cnt'].item()>=min(cnt_thres,idx))]

    relation_set = set(relation_list)  # remove duplicate relations

    repeated_idx = []
    relation_array = np.array(list(relation_set))
    for i in range(len(relation_array)):
        for j in range(len(relation_array)):
            res = relation_array[i] == relation_array[j]
            if res[0] and res[2]and i!=j:
                repeated_idx.append(i)
    repeated_idx = set(repeated_idx)
    repeated_idx = list(repeated_idx)
    if len(repeated_idx)>0:
        repeated = relation_array[repeated_idx]
        #print repeated.shape, repeated_idx
        for i, (pos, x, y) in enumerate(zip(repeated_idx, repeated[:, 0], repeated[:, 2])):
            position = np.where((x == repeated[:, 0]) & (y == repeated[:, 2]))[0]
            triplets = repeated[position].astype(int).tolist()
            preds = [t[1] for t in triplets]
            counted = Counter(preds)
            voted_pred = counted.most_common(1)
            #print(i, idx, triplets, voted_pred)
            relation_array[pos, 1] = voted_pred[0][0]

        relation_set =[tuple(rel)for rel in relation_array.astype(int).tolist()]
        relation_set = set(relation_set)
        #print(len(relation_set))

    for rel_num in range(len(relation_set)):
        rel = relation_set.pop()
        tile = False
        handle = False
        for t_i in tile_idx:
            if (str(rel[0]) == t_i or str(rel[2]) == t_i):
                tile = True
        for h_i in handle_idx:
            if (str(rel[0]) == h_i or str(rel[2]) == h_i):
                handle = True
        if ( (not tile) and (not handle)):
            sg.edge('struct'+str(rel[0])+':'+str(rel[0]),
                    'struct'+str(rel[2])+':'+'-'+str(rel[2]),
                    str(test_set.predicate_classes[rel[1]]))
    sg.render(osp.join(save_path,'scene_graph'+str(idx)), view=view)
    node_feature.to_json(osp.join(save_path,'json','scene_graph_node'+str(idx)+'.json'), orient = 'index')
    relation.to_json(osp.join(save_path,'json','scene_graph_relation'+str(idx)+'.json'), orient = 'index')
    #sg.clear()


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def get_color_hist(img):
    '''
    # return color_hist
    # format: [[num_pixels1,color1],[num_pixels2,color2],...,[num_pixelsN,colorN]]
    # ex:     [[362        ,'red' ],[2          ,'blue'],...,[3          ,'gray']]
    '''

    img = img[..., ::-1]  # BGR to RGB
    img = img.flatten().reshape(-1, 3).tolist()  # shape: ((640x480)*3)

    color_hist = []
    start = 0
    new_color = False
    actual_name, closest_name = get_colour_name(img[0])
    if (actual_name == None):
        color_hist.append([0,closest_name])
    else:
        color_hist.append([0,actual_name])

    for i in range(len(img)):
        actual_name, closest_name = get_colour_name(img[i])
        for k in range(len(color_hist)):
            if(color_hist[k][1] == actual_name or color_hist[k][1] == closest_name):
                color_hist[k][0]+=1
                new_color = False
                break
            else:
                new_color = True
        if (new_color == True):         
            if (actual_name == None):
                color_hist.append([1, closest_name])
                new_color = False
            else:
                color_hist.append([1, actual_name])
                new_color = False
    color_hist = sorted(color_hist, reverse = True)
    return color_hist


def isNoisyPoint(point,range=10.0):
    return -range< point[0]<range and -range<point[1]<range and -range<point[2]<range


def outlier_filter(points,mean_k=10,thres=1.0):
    try:
        points_ = np.array(points, dtype=np.float32)
        cloud = pcl.PointCloud()
        cloud.from_array(points_)
        filtering = cloud.make_statistical_outlier_filter()
        filtering.set_mean_k(min(len(points_), mean_k))
        filtering.set_std_dev_mul_thresh(thres)
        cloud_filtered = filtering.filter()
        return cloud_filtered.to_array().tolist()
    except:
        return points



def get_color_hist2(img):
    '''
    # return color_hist
    # format: [[density1,color1],[density2,color2],[density3,color3]]
    # ex:     [[362     ,'red' ],[2       ,'blue'],[3       ,'gray']]
    '''
    try:
        hist3D = Hist3D(img[..., ::-1], num_bins=8, color_space='rgb')# BGR to RGB
        # print('sffsd:', img.shape)
        # cv2.imshow('a',img)
        # cv2.waitKey(1)
    except:

        return get_color_hist(img)
    else:
        densities = hist3D.colorDensities()
        order = densities.argsort()[::-1]
        densities = densities[order]
        colors = (255*hist3D.rgbColors()[order]).astype(int)
        color_hist = []
        for density, color in zip(densities,colors)[:4]:
            actual_name, closest_name = get_colour_name(color.tolist())
            if (actual_name == None):
                color_hist.append([density, closest_name])
            else:
                color_hist.append([density, actual_name])

    return color_hist

def make_window_size(width, height, obj_boxes):
    if( width<30):
        range_x_min = int(obj_boxes[0]) + int(width*2./10.) 
        range_x_max = int(obj_boxes[0]) + int(width*8./10.)
    elif(width < 60):
        range_x_min = int(obj_boxes[0]) + int(width*3./20.)
        range_x_max = int(obj_boxes[0]) + int(width*17./20.)
    else:
        range_x_min = int(obj_boxes[0]) + int(width*4./30.)
        range_x_max = int(obj_boxes[0]) + int(width*26./30.)
        
    if (height < 30):
        range_y_min = int(obj_boxes[1]) + int(height*2./10.)
        range_y_max = int(obj_boxes[1]) + int(height*8./10.)
    elif (height < 60):
        range_y_min = int(obj_boxes[1]) + int(height*3./20.)
        range_y_max = int(obj_boxes[1]) + int(height*17./20.)
    else:
        range_y_min = int(obj_boxes[1]) + int(height*4./30.)
        range_y_max = int(obj_boxes[1]) + int(height*26./30.)

    return range_x_min, range_x_max, range_y_min, range_y_max

colorlist = [(random.randint(0,230),random.randint(0,230),random.randint(0,230)) for i in range(10000)]

class scene_graph(object):
    def __init__(self,args):
        #self.data = DataFrame({"node_feature":[]}, index=[])
        # [class, index, score, bounding_box, 3d_pose, mean, var, pt_number, color]
        
        self.data = DataFrame({"class":[np.zeros(400)], "idx":0, "score":[np.zeros(400)], #check
                                "bounding_box":[[0,0,0,0]], "3d_pose":[[0,0,0]], "mean":[[0,0,0]],
                                "var":[[0,0,0]], "pt_num":0, "color_hist":[[[0,"red"],[0,"blue"]]], "detection_cnt:":0},
                                columns = ['class','idx','score','bounding_box','3d_pose','mean',
                                'var','pt_num','color_hist','detection_cnt'])
        self.rel_data = DataFrame({"relation":[]}, index=[])
        self.img_count = 0
        self.pt_num = 0
        self.mean = [0,0,0]
        self.var = [0,0,0]
        self.args = args
        self.detect_cnt_thres = args.detect_cnt_thres
        self.fig=plt.figure()
        scene_name = args.scannet_path.split('/')[-1]
        self.save_path = osp.join(self.args.vis_result_path,scene_name,'scene_graph')
        try: os.makedirs(self.save_path)
        except: pass
        try: os.makedirs(osp.join(self.save_path, 'json'))
        except: pass
        self.disable_samenode = self.args.disable_samenode
        if self.disable_samenode: self.detect_cnt_thres=0

    def vis_scene_graph(self, image_scene, idx, test_set, obj_inds, obj_boxes, obj_scores,
                        subject_inds, predicate_inds, object_inds,
                        subject_IDs, object_IDs, triplet_scores,relationships,
                        pix_depth=None, inv_p_matrix=None, inv_R=None, Trans=None, dataset ='scannet'):
        updated_image_scene = image_scene.copy()
        sg = Digraph('structs', node_attr = {'shape': 'plaintext'}) # initialize scene graph tool
        if dataset == 'scannet':   #scannet
            print('-ID--|----Object-----|Score|3D_position (x, y, z)|---var-------------|---color------')
        else:
            print('-ID--|----Object-----|---Score---------------------------------------------')


        ax = self.fig.add_subplot(111, projection='3d')
        #print('sfdsfdfsd:',obj_boxes.shape)
        for i, obj_ind in enumerate(obj_inds):  # loop for bounding boxes on each images

            if dataset == 'scannet':

                '''1. Get Color Histogram'''
                # color_hist
                # ex: [[num_pixels1,color1],[num_pixels2,color2],...,[num_pixelsN,colorN]]
                #     [[362        ,'red' ],[2          ,'blue'],...,[3          ,'gray']]
                box_whole_img = image_scene[int(obj_boxes[i][1]):int(obj_boxes[i][3]),
                                int(obj_boxes[i][0]):int(obj_boxes[i][2])]
                color_hist = get_color_hist2(box_whole_img)


                '''2. Get Center Patch '''
                # Define bounding box info
                width = int(obj_boxes[i][2]) - int(obj_boxes[i][0])
                height = int(obj_boxes[i][3]) - int(obj_boxes[i][1])
                box_center_x = int(obj_boxes[i][0]) + width/2
                box_center_y = int(obj_boxes[i][1]) + height/2
                # using belows to find mean and variance of each bounding boxes
                # pop 1/5 size window_box from object bounding boxes 
                range_x_min, range_x_max, range_y_min, range_y_max = make_window_size(width, height, obj_boxes[i])
                # Crop center patch
                box_center_img = image_scene[range_y_min:range_y_max, range_x_min:range_x_max]






                '''3. Get 3D positions of the Centor Patch'''
                window_3d_pts = []
                for pt_x in range(range_x_min, range_x_max):
                    for pt_y in range(range_y_min, range_y_max):
                        pose_2d_window = np.matrix([pt_x, pt_y, 1])
                        pose_3d_window = pix_depth[pt_x][pt_y] * np.matmul(inv_p_matrix, pose_2d_window.transpose())
                        pose_3d_world_coord_window = np.matmul(inv_R, pose_3d_window[0:3] - Trans.transpose())
                        if not isNoisyPoint(pose_3d_world_coord_window):
                            # save several points in window_box to calculate mean and variance
                            window_3d_pts.append([pose_3d_world_coord_window.item(0), pose_3d_world_coord_window.item(1), pose_3d_world_coord_window.item(2)])
                # window_3d_pts
                # ex: [[X_1,Y_1,Z_1],[X_2,Y_2,Z_2],...,[X_N,Y_N,Z_N]]

                # window_3d_pts = []
                # for pt_x in range(int(obj_boxes[i][0]), int(obj_boxes[i][2])):
                #     for pt_y in range(int(obj_boxes[i][1]), int(obj_boxes[i][3])):
                #         pose_2d_window = np.matrix([pt_x, pt_y, 1])
                #         pose_3d_window = pix_depth[pt_x][pt_y] * np.matmul(inv_p_matrix, pose_2d_window.transpose())
                #         pose_3d_world_coord_window = np.matmul(inv_R, pose_3d_window[0:3] - Trans.transpose())
                #         if not isNoisyPoint(pose_3d_world_coord_window):
                #             # save several points in window_box to calculate mean and variance
                #             window_3d_pts.append([pose_3d_world_coord_window.item(0), pose_3d_world_coord_window.item(1), pose_3d_world_coord_window.item(2)])


                window_3d_pts = outlier_filter(window_3d_pts)

                #window_3d_pts = np.array(window_3d_pts,dtype=np.float32)
                #cloud = pcl.PointCloud()
                #cloud.from_array(window_3d_pts)
                #outlier_filter = cloud.make_statistical_outlier_filter()
                #outlier_filter.set_mean_k(min(len(window_3d_pts),10))
                #outlier_filter.set_std_dev_mul_thresh(1.0)
                #cloud_filtered = outlier_filter.filter()
                #window_3d_pts = cloud_filtered.to_array().tolist()

                # arr = np.array(window_3d_pts,dtype=np.float).reshape(-1,3)
                # if arr.size>0:
                #     ax.scatter(-arr[:,0],-arr[:,1],-arr[:,2],)
                #     #ax.set_xlim(-2000, 2000)
                #     #ax.set_ylim(-2000, 2000)
                #     #ax.set_zlim(-2000, 2000)
                #
                #     self.fig.show()
                #     plt.pause(0.01)
                #     plt.hold(True)
                # cv2.waitKey(0)





                '''4. Get a 3D position of the Center Patch's Center point'''
                # find 3D point of the bounding box(the center patch)'s center
                curr_pt_num, curr_mean, curr_var = Measure_new_Gaussian_distribution(window_3d_pts)
                # ex: np.matrix([[X_1],[Y_1],[Z_1]])

                # get object class names as strings
                box_cls = [test_set.object_classes[obj_ind[0]], 
                           test_set.object_classes[obj_ind[1]],
                           test_set.object_classes[obj_ind[2]]]
                # box_cls: ['pillow','bag','cat']
                box_score = obj_scores[i]
                # box_score: [0.2,0.1,0.01]
                cls_scores = np.zeros(400)
                for cls_idx, cls_score in zip(obj_ind, obj_scores[i]):
                    cls_scores[cls_idx] += cls_score  # check


                '''5. Save Object Recognition Results in DataFrame Format'''
                if(self.img_count ==0):
                    # first image -> make new node
                    box_id = i
                    self.pt_num, self.mean, self.var = Measure_new_Gaussian_distribution(window_3d_pts)
                    # check
                    start_data = {"class":cls_scores, "idx":box_id, "score":box_score,
                                  "bounding_box":[box_center_x,box_center_y,width,height],
                                  "3d_pose": [int(self.mean[0]),int(self.mean[1]),int(self.mean[2])],
                                  "mean":self.mean,
                                  "var":self.var,
                                  "pt_num":self.pt_num,
                                  "color_hist":color_hist,
                                  "detection_cnt":1
                                  }
                    obj_boxes[i][4] =box_id
                    self.data.loc[len(self.data)] = start_data
                    if (i==0):                        
                        self.data.drop(self.data.index[0], inplace=True)
                        self.data.rename(index={1:0}, inplace=True)
                    #print(self.data)

                else:
                    # get node similarity score
                    node_score, max_score_index = node_update(window_3d_pts, self.data, curr_mean,curr_var,
                                                              box_cls, obj_scores[i], color_hist,test_set)
                    threshold = 0.8127
                    
                    if node_score > threshold and not self.disable_samenode:
                        # change value of global_node
                        # change global_node[max_score_index]
                        print("node updated!!!")
                        for cls_idx,cls_score in zip(obj_ind,obj_scores[i]):
                            self.data.at[max_score_index,'class'][cls_idx]+= cls_score # check

                        #self.data.at[max_score_index, "class"] = box_cls
                        self.data.at[max_score_index, "score"] = node_score
                        self.pt_num, self.mean, self.var = Measure_added_Gaussian_distribution(window_3d_pts,
                                                                                self.data.ix[max_score_index]["mean"],
                                                                                self.data.ix[max_score_index]["var"],
                                                                                self.data.ix[max_score_index]["pt_num"],
                                                                                len(window_3d_pts)) 
                        self.data.at[max_score_index, "mean"] = self.mean
                        self.data.at[max_score_index, "var"] = self.var
                        self.data.at[max_score_index, "pt_num"] = self.pt_num
                        self.data.at[max_score_index, "color_hist"] = color_hist
                        self.data.at[max_score_index, "detection_cnt"] = self.data.ix[max_score_index]["detection_cnt"]+1
                        box_id = self.data.ix[max_score_index]["idx"]
                        obj_boxes[i][4] = box_id
                    else:
                        # make new_node in global_node
                        # [class, index, score, bounding_box, 3d_pose, mean, var, pt_number, color_hist]
                        box_id = len(self.data)+1
                        obj_boxes[i][4] = box_id
                        self.pt_num, self.mean, self.var = Measure_new_Gaussian_distribution(window_3d_pts)
                        global_node_num = len(self.data)
                        add_node_list = [cls_scores, box_id, box_score, [box_center_x, box_center_y, width, height],
                                         [self.mean[0], self.mean[1], self.mean[2]],
                                         self.mean, self.var, self.pt_num, color_hist,1]
                        self.data.loc[len(self.data)] = add_node_list

                # if object index was changed, update relation's object index also


                '''6. Print object info'''
                print('{obj_ID:5} {obj_cls:15}  {obj_score:4.2f} {object_3d_pose:20}    {obj_var:20} {obj_color:15}'
                          .format(obj_ID= box_id, 
                                  obj_cls= box_cls[0],
                                  obj_score= box_score[0],
                                  object_3d_pose= [self.mean[0], self.mean[1], self.mean[2]],
                                  obj_var= self.var,
                                  obj_color = color_hist[0][1] ))


            else:   # TODO: for visual_genome
                raise NotImplementedError

            '''7. Plot '''
            # updated object_detection
            cv2.rectangle(updated_image_scene,
                          (int(obj_boxes[i][0]), int(obj_boxes[i][1])),
                          (int(obj_boxes[i][2]), int(obj_boxes[i][3])),
                          colorlist[int(obj_boxes[i][4])],
                          2)
            font_scale=0.5
            txt = str(box_id) + '. ' + str(box_cls[0]) + ' ' + str(round(box_score[0],2))
            ((txt_w, txt_h), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # Place text background.
            x0, y0 = int(obj_boxes[i][0]),int(obj_boxes[i][3])
            back_tl = x0, y0 - int(1.3 * txt_h)
            back_br = x0 + txt_w, y0
            cv2.rectangle(updated_image_scene, back_tl, back_br, colorlist[int(obj_boxes[i][4])], -1)
            cv2.putText(updated_image_scene,
                        txt,
                        (x0,y0-2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255,255,255),
                        1)

        # add ID per bbox

        rel_prev_num = len(self.rel_data)
        print('-------Subject--------|-------Predicate-----|--------Object---------|--Score-')
        for i, relation in enumerate(relationships):
            # update relation's class also

            # accumulate relation_list
            if str(int(obj_boxes[int(relation[0])][4])) != str(int(obj_boxes[int(relation[1])][4])):
            # filter out triplets whose sbj == obj
                self.rel_data.loc[len(self.rel_data)] = [[str(int(obj_boxes[int(relation[0])][4])), int(relation[2]), str(int(obj_boxes[int(relation[1])][4]))]]

                print('{sbj_cls:9} {sbj_ID:4} {sbj_score:1.3f}  |  '
                      '{pred_cls:11} {pred_score:1.3f}  |  '
                      '{obj_cls:9} {obj_ID:4} {obj_score:1.3f}  |  '
                      '{triplet_score:1.3f}'.format(
                    sbj_cls = test_set.object_classes[obj_inds[:,0][int(relation[0])]], sbj_score = obj_scores[:,0][int(relation[0])],
                    sbj_ID = str(int(obj_boxes[int(relation[0])][4])),
                    pred_cls = test_set.predicate_classes[int(relation[2])] , pred_score = relation[3] / obj_scores[:,0][int(relation[0])] / obj_scores[:,0][int(relation[1])],
                    obj_cls = test_set.object_classes[obj_inds[:,0][int(relation[1])]], obj_score = obj_scores[:,0][int(relation[1])],
                    obj_ID = str(int(obj_boxes[int(relation[1])][4])),
                    triplet_score = relation[3]))

        rel_new_num = len(self.rel_data)

        # Draw scene graph
        if ( rel_prev_num != rel_new_num): 
            Draw_connected_scene_graph(self.data, self.rel_data, self.img_count, test_set, sg, idx,
                                       self.detect_cnt_thres,self.args.plot_graph,self.save_path)
        #sg.view()

        # it's help to select starting point of first image manually
        self.img_count+=1

        return updated_image_scene


def vis_object_detection(image_scene,test_set,
                    obj_inds,obj_boxes,obj_scores):

    for i, obj_ind in enumerate(obj_inds):
        cv2.rectangle(image_scene,
                      (int(obj_boxes[i][0]), int(obj_boxes[i][1])),
                      (int(obj_boxes[i][2]), int(obj_boxes[i][3])),
                      colorlist[int(obj_boxes[i][4])],
                      2)
        font_scale=0.5
        txt =str(int(obj_boxes[i][4])) + '. '+get_class_string(obj_ind,obj_scores[i],test_set.object_classes)
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

    return image_scene
