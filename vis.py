import cv2
import random
import numpy as np
from pandas import DataFrame
import pandas as pd
from graphviz import Digraph
import webcolors
import pprint
import math
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

def Point_and_Score_based_object_correction(window_3d_pts, img_count, node_feature, pose_3d_world_coord, box_cls, box_id, box_score):
    pt_num, mean, var  = 0, [0,0,0], [0,0,0]
    if(img_count == 0):
        # First image just calculate new mean and variance
        pt_num, mean, var = Measure_new_Gaussian_distribution(window_3d_pts)
    else :
        # If another images come in
        case = -1                           # it's used to whether make new objects or not
        new_pt_num = len(window_3d_pts)     
        pt_updated = False                  # it's used to break for loop if point is updated
        for im_num in range(img_count):     
            if(pt_updated == True):         # if point was updated, stop searching previous boxes
                break
            im_num = img_count-1-im_num     # To match with recently updated boxes
            for box_num in range(len(node_feature[im_num])):
                prev_mean = node_feature[im_num][box_num][5]
                prev_var = node_feature[im_num][box_num][6]
                prev_pt_num = node_feature[im_num][box_num][7]
                # Judge whether current box's points are contained on previous boxes or not by using Gaussian probability
                x_check, y_check, z_check = Gaussian_prob([int(pose_3d_world_coord.item(0)),
                                                           int(pose_3d_world_coord.item(1)),
                                                           int(pose_3d_world_coord.item(2))], prev_mean, prev_var)
                if(box_cls == str(node_feature[im_num][box_num][0])):       # check object's class first
                    if(box_id == str(node_feature[im_num][box_num][1])):    # find same class & id
                        if(x_check & y_check & z_check):                    #   - that's position is similar with current also
                            case = 0
                            # update mean and variance of prev_detected box
                            pt_num, mean, var = Measure_added_Gaussian_distribution(window_3d_pts, prev_mean, prev_var, prev_pt_num, new_pt_num)
                            pt_updated = True   
                            print(box_cls + " Point is updated")
                            break           # break double for loop
                        else:                                               #   - position is different
                            case = 1
                            pt_num = prev_pt_num
                            mean = prev_mean
                            var = prev_var
                            pt_updated = True
                            # break double for loop without update previous box
                            break
                    else:                                                   # find same class, but different id
                        if(x_check & y_check & z_check):                    #   - position is similar
                                                                            #     --> same objects but id was additionally generated
                            case = 2
                            # update mean and variance of prev_detected box
                            pt_num, mean, var = Measure_added_Gaussian_distribution(window_3d_pts, prev_mean, prev_var, prev_pt_num, new_pt_num)
                            pt_updated = True
                            print("Class:"+box_cls + " id was updated "+ box_id+ " to " +str(node_feature[im_num][box_num][1]))
                            # change current box_id to previous detected box id
                            box_id = str(node_feature[im_num][box_num][1])   
                            break
                        else:                                               #   - position is different
                                                                            #     --> New objects    
                            case = 3
                            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                else:                                                       # different box class
                    if(box_id == str(node_feature[im_num][box_num][1])):    # find same id, but different class
                        case = 4
                        # Compare objects score to confirm class
                        if(box_score < node_feature[im_num][box_num][2]):
                            print("ID:"+box_id + " class was updated "+ box_cls + " to " + str(node_feature[im_num][box_num][0]))
                            # if previous object class's score are higher than current, update class name and score
                            box_cls = str(node_feature[im_num][box_num][0])
                            box_score = node_feature[im_num][box_num][2]
                        # same id --> it's maybe same object --> update position
                        pt_num, mean, var = Measure_added_Gaussian_distribution(window_3d_pts, prev_mean, prev_var, prev_pt_num, new_pt_num)
                        pt_updated = True
                        break
                    else:                                                   # different id & class --> New objects
                        case = 5
                        
        if(case == 3 or case == 5):     # make new object's mean and variance
            print("new objects")
            pt_num, mean, var = Measure_new_Gaussian_distribution(window_3d_pts)
    
    return pt_num, mean, var, box_cls, box_id, box_score

def Gaussian_prob(x, mean, var):
    Z_x = (x[0]-mean[0])/np.sqrt(var[0])
    Z_y = (x[1]-mean[1])/np.sqrt(var[1])
    Z_z = (x[2]-mean[2])/np.sqrt(var[2])
    # In Standardized normal gaussian distribution
    # Threshold : 0.9 --> -1.65 < Z < 1.65
    #           : 0.8 --> -1.29 < Z < 1.29
    #           : 0.7 --> -1.04 < Z < 1.04
    threshold = 1.04
    x_check = -threshold < Z_x < threshold
    y_check = -threshold < Z_y < threshold
    z_check = -threshold < Z_z < threshold
    return x_check, y_check, z_check

def Measure_new_Gaussian_distribution(new_pts):
    pt_num = len(new_pts)
    mu = np.sum(new_pts, axis=0)/pt_num
    mean = [int(mu[0]), int(mu[1]), int(mu[2])]
    var = np.sum(np.power(new_pts, 2), axis=0)/pt_num - np.power(mu,2)
    var = [int(var[0]), int(var[1]), int(var[2])]
    return pt_num, mean, var

def Measure_added_Gaussian_distribution(new_pts, prev_mean, prev_var, prev_pt_num, new_pt_num):
    # update mean and variance
    pt_num = prev_pt_num + new_pt_num
    mu = np.sum(new_pts, axis=0)
    mean = np.divide((np.multiply(prev_mean,prev_pt_num) + mu),pt_num)
    mean = [int(mean[0]), int(mean[1]), int(mean[2])]
    var = np.subtract(np.divide((np.multiply((prev_var + np.power(prev_mean,2)),prev_pt_num) + np.sum(np.power(new_pts,2),axis=0)) ,pt_num), np.power(mean,2))
    var = [int(var[0]), int(var[1]), int(var[2])]
    return pt_num, mean, var

def Draw_connected_scene_graph(node_feature, img_count, relation, test_set, sg, idx):
    # load all of saved node_feature
    # if struct ids are same, updated to newly typed object
    for img_num in range(len(node_feature)):
        for i in range(len(node_feature[img_num])):
            if len(node_feature[img_num][i][8]) ==1:
                node_feature[img_num][i][8].append(node_feature[img_num][i][8][0])
            box_color_bgr = colorlist[int(node_feature[img_num][i][1])]
            box_color_rgb = box_color_bgr[::-1]
            box_color_hex = webcolors.rgb_to_hex(box_color_rgb)
            sg.attr('node', style="", color=box_color_hex)
            sg.node('struct'+str(node_feature[img_num][i][1]), '''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
              <TR>
                <TD PORT="'''+str(node_feature[img_num][i][1])+'''" ROWSPAN="2">'''+str(node_feature[img_num][i][0])+'''</TD>
                <TD COLSPAN="1">ID</TD>
                <TD COLSPAN="2">color</TD>
                <TD COLSPAN="3">3D Position</TD>
              </TR>
              <TR>
                <TD>'''+str(node_feature[img_num][i][1])+'''</TD>
                <TD>'''+str(node_feature[img_num][i][8][0][1])+'''</TD>
                <TD>'''+str(node_feature[img_num][i][8][1][1])+'''</TD>
                <TD>'''+str(node_feature[img_num][i][4][0]) + '''</TD>
                <TD>'''+str(node_feature[img_num][i][4][1])+'''</TD>
                <TD>'''+str(node_feature[img_num][i][4][2])+'''</TD>
              </TR>
            </TABLE>>''')

    relation_list = []
    for img_num in range(len(relation)):    
        for j in range(len(relation[img_num])):
            relation_list.append(relation[img_num][j])

    relation_set = set(relation_list)  # remove duplicate relations
    for rel_num in range(len(relation_set)):
        rel = relation_set.pop()
        sg.edge('struct'+str(rel[0])+':'+str(rel[0]),
                'struct'+str(rel[2])+':'+str(rel[2]),
                 str(test_set.predicate_classes[rel[1]]))

    '''
    for rel_num in range(len(relation_list)):
        sg.edge('struct'+str(relation_list[rel_num][0])+':'+str(relation_list[rel_num][0]),
                'struct'+str(relation_list[rel_num][2])+':'+str(relation_list[rel_num][2]),
                 str(test_set.predicate_classes[relation_list[rel_num][1]]))
    '''
    sg.render('scene_graph_test/scene-graph_'+str(idx)+'.gv.pdf', view=True, cleanup=False)
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


colorlist = [(random.randint(0,230),random.randint(0,230),random.randint(0,230)) for i in range(10000)]

class scene_graph(object):
    def __init__(self):
        self.data = DataFrame({"node_feature":[]}, index=[])
        self.data["relation"] = []
        self.img_count = 0
        self.pt_num = 0
        self.mean = [0,0,0]
        self.var = [0,0,0]

    def vis_scene_graph(self, image_scene, idx, test_set, obj_inds, obj_boxes, obj_scores, 
                        subject_inds, predicate_inds, object_inds,
                        subject_IDs, object_IDs, triplet_scores,
                        pix_depth=None, inv_p_matrix=None, inv_R=None, Trans=None, dataset ='scannet'):
        ''' 5. Print Scene Graph '''
        sg = Digraph('structs', node_attr = {'shape': 'plaintext'}) # initialize scene graph tool
        if dataset == 'scannet':   #scannet
            print('-ID--|----Object-----|---Score---|---3d_position (x, y, z)-------|---mean-----------|---var------------------')
        else:
            print('-ID--|----Object-----|---Score---------------------------------------------')
        node_feature_list = []
        for i, obj_ind in enumerate(obj_inds):  # loop for bounding boxes on each images
            window_3d_pts = []
            if dataset == 'scannet':
                width = int(obj_boxes[i][2]) - int(obj_boxes[i][0])
                height = int(obj_boxes[i][3]) - int(obj_boxes[i][1])
                box_center_x = int(obj_boxes[i][0]) + width/2
                box_center_y = int(obj_boxes[i][1]) + height/2

                '''
                box_img = image_scene[int(obj_boxes[i][1]):int(obj_boxes[i][3]), int(obj_boxes[i][0]):int(obj_boxes[i][2])]
                box_img = box_img.flatten().reshape(-1,3).tolist()
                color_hist = get_color_hist(box_img)
                print("get histogram of obj color")
                '''

                # using belows to find mean and variance of each bounding boxes
                # pop 1/5 size window_box from object bounding boxes 
                range_x_min = int(obj_boxes[i][0]) + int(width*2/5) 
                range_x_max = int(obj_boxes[i][0]) + int(width*3/5)
                range_y_min = int(obj_boxes[i][1]) + int(height*2/5)
                range_y_max = int(obj_boxes[i][1]) + int(height*3/5)

                box_img = image_scene[range_y_min:range_y_max, range_x_min:range_x_max]
                box_img = box_img[...,::-1] # BGR to RGB
                box_img = box_img.flatten().reshape(-1,3).tolist()
                color_hist = get_color_hist(box_img)

                for pt_x in range(range_x_min, range_x_max):
                    for pt_y in range(range_y_min, range_y_max):
                        pose_2d_window = np.matrix([pt_x, pt_y, 1])
                        pose_3d_window = pix_depth[pt_x][pt_y] * np.matmul(inv_p_matrix, pose_2d_window.transpose())
                        pose_3d_world_coord_window = np.matmul(inv_R, pose_3d_window[0:3] - Trans.transpose())
                        # save several points in window_box to calculate mean and variance
                        window_3d_pts.append([pose_3d_world_coord_window.item(0), pose_3d_world_coord_window.item(1), pose_3d_world_coord_window.item(2)])
                
                # find 3D point of bounding box's center
                pose_2d = np.matrix([box_center_x, box_center_y, 1])
                pose_3d = pix_depth[box_center_x][box_center_y] * np.matmul(inv_p_matrix, pose_2d.transpose())
                pose_3d_world_coord = np.matmul(inv_R, pose_3d[0:3] - Trans.transpose())
                
                # update current bounding box using Point_and_Score_based_object_correction function
                self.pt_num, self.mean, self.var, box_cls, box_id, box_score = Point_and_Score_based_object_correction(window_3d_pts, self.img_count, self.data.node_feature, pose_3d_world_coord, test_set.object_classes[obj_ind], str(int(obj_boxes[i][4])), obj_scores[i])
                
                # if object index was changed, update relation's object index also
                if( box_id != str(int(obj_boxes[i][4])) ):  # box_id : updated id, right_side : original box index
                    for j in range(len(predicate_inds)):
                        if predicate_inds[j] > 0:
                            if ( str(int(obj_boxes[i][4])) == str(subject_IDs[j]) ):    # find relation's index that same as original box index
                                subject_IDs[j] = box_id                                 # update relation index
                            if ( str(int(obj_boxes[i][4])) == str(object_IDs[j]) ):
                                object_IDs[j] = box_id
                
                if( len(color_hist) >1):
                    print('{obj_ID:5} {obj_cls:15} {obj_score:10.3f} {object_3d_pose:33} {obj_mean:15} {obj_var:15} {obj_color:15}'
                              .format(obj_ID= box_id, 
                                      obj_cls= box_cls, 
                                      obj_score= box_score,
                                      object_3d_pose= (pose_3d_world_coord.transpose().round()).tolist(), 
                                      obj_mean= self.mean, 
                                      obj_var= self.var,
                                      obj_color = [[color_hist[0][1],color_hist[0][0]],[color_hist[1][1],color_hist[1][0]]] ))
                else:
                    print('{obj_ID:5} {obj_cls:15} {obj_score:10.3f} {object_3d_pose:33} {obj_mean:15} {obj_var:15} {obj_color:15}'
                              .format(obj_ID= box_id, 
                                      obj_cls= box_cls, 
                                      obj_score= box_score,
                                      object_3d_pose= (pose_3d_world_coord.transpose().round()).tolist(), 
                                      obj_mean= self.mean, 
                                      obj_var= self.var,
                                      obj_color = [color_hist[0][1],color_hist[0][0]] ))

                # [class, index, score, bounding_box, 3d_pose, mean, var, pt_number]
                node_feature_list.append([box_cls,
                                        box_id,
                                        box_score,
                                        [box_center_x, box_center_y, width, height],
                                        [int(pose_3d_world_coord.item(0)),
                                        int(pose_3d_world_coord.item(1)),
                                        int(pose_3d_world_coord.item(2))],
                                        self.mean,
                                        self.var,
                                        self.pt_num,
                                        color_hist])

            else:   # for vis_genome
                print('{obj_ID:5} {obj_cls:15} {obj_score:1.3f}'
                      .format(obj_ID=str(int(obj_boxes[i][4])), obj_cls=test_set.object_classes[obj_ind], obj_score=obj_scores[i]))
                node_feature_list.append([test_set.object_classes[obj_ind],
                                        str(int(obj_boxes[i][4])),
                                        [0, 0, 0, 0],
                                        [0,0,0]])

            # object_detection
            cv2.rectangle(image_scene,
                          (int(obj_boxes[i][0]), int(obj_boxes[i][1])),
                          (int(obj_boxes[i][2]), int(obj_boxes[i][3])),
                          colorlist[int(obj_boxes[i][4])],
                          2)
            font_scale=0.5
            txt = str(box_id) + '. ' + str(box_cls) + ' ' + str(round(box_score,2))
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
        rows.append(node_feature_list) 

        relation_list = []
        print('-------Subject-------|------Predicate-----|--------Object---------|--Score-')
        for i in range(len(predicate_inds)):
            if predicate_inds[i] > 0: # predicate_inds[i] = 0 is the class 'irrelevant'
                # update relation's class also
                for j in range(len(node_feature_list)):
                    if(str(subject_IDs[i]) == str(node_feature_list[j][1]) ): # find relation's object id that same as bounding box id on current image
                        sbj_cls_input = node_feature_list[j][0]               # update relation's class to updated object class
                    if(str(object_IDs[i]) == str(node_feature_list[j][1]) ):
                        obj_cls_input = node_feature_list[j][0]

                print('{sbj_cls:9} {sbj_ID:4} {sbj_score:1.2f}  |  '
                      '{pred_cls:11} {pred_score:1.2f}  |  '
                      '{obj_cls:9} {obj_ID:4} {obj_score:1.2f}  |  '
                      '{triplet_score:1.3f}'.format(
                    sbj_cls = sbj_cls_input, sbj_score = triplet_scores[i][0],
                    sbj_ID = str(subject_IDs[i]),
                    pred_cls = test_set.predicate_classes[predicate_inds[i]] , pred_score = triplet_scores[i][1],
                    obj_cls = obj_cls_input, obj_score = triplet_scores[i][2],
                    obj_ID = str(object_IDs[i]),
                    triplet_score = np.prod(triplet_scores[i])))
                # accumulate relation_list
                relation_list.append((str(subject_IDs[i]), predicate_inds[i], str(object_IDs[i])))
        
        # Add relation info to DataFrame
        rows.append(relation_list)
        self.data.loc[len(self.data)] = rows
        self.data = self.data.rename(index={idx:"img"+str(idx)})

        '''
        data.node_feature[0] : image number select
        data.node_feature[0][0] : show one pack of object node info
        data.node_feature[0][0][0] : select node's info [0~7] (class, id, score, bounding_box, 3d_pose, mean, var, pt_num)

        data.relation[0] : image number select
        data.relation[0][0] : show one pack of object relation info
        data.relation[0][0][0] : select relation's info [0~2] (subject, relation, object)
        '''
        # Draw scene graph
        #Draw_connected_scene_graph(self.data.node_feature, self.img_count, self.data.relation, test_set, sg, idx)
        #sg.view()

        # it's help to select starting point of first image manually
        self.img_count+=1

        return image_scene

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

def vis_caption(image_caption,test_set,region_caption,region_boxes,region_scores):
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
                          2)
            font_scale = 0.5
            txt = str(i) + '. ' + sentence + ' ' + str(region_scores[i])[:4]
            ((txt_w, txt_h), _) = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            # Place text background.
            x0, y0 = int(region_boxes[i][0]),int(region_boxes[i][3])
            back_tl = x0, y0 - int(1.3 * txt_h)
            back_br = x0 + txt_w, y0
            cv2.rectangle(image_caption, back_tl, back_br, colorlist[i], -1)
            cv2.putText(image_caption,
                        txt,
                        (x0, y0-2 ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        1)
    return image_caption
