import cv2
import random
import numpy as np
from pandas import DataFrame
import pandas as pd
from graphviz import Digraph
import webcolors
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



colorlist = [(random.randint(0,230),random.randint(0,230),random.randint(0,230)) for i in range(10000)]

class scene_graph(object):
    def __init__(self):
        self.data = DataFrame({"node_feature":[]}, index=[])
        self.data["relation"] = []
        self.img_count = 0
    def vis_scene_graph(self, idx, test_set, obj_inds, obj_boxes, obj_scores, 
                        subject_inds, predicate_inds, object_inds,
                        subject_IDs, object_IDs, triplet_scores,
                        pix_depth=None, inv_p_matrix=None, inv_R=None, Trans=None, dataset ='scannet'):
        ''' 5. Print Scene Graph '''
        sg = Digraph('structs', node_attr = {'shape': 'plaintext'})
        if dataset == 'scannet':   #scannet
            print('-ID--|----Object-----|---Score---|---3d_position (x, y, z)-----------------')
        else:
            print('-ID--|----Object-----|---Score---------------------------------------------')
        node_feature_list = []
        for i, obj_ind in enumerate(obj_inds):
            if dataset == 'scannet':
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
            
                node_feature_list.append([test_set.object_classes[obj_ind],
                                        str(int(obj_boxes[i][4])),
                                        [box_center_x, box_center_y, width, height],
                                        [int(pose_3d_world_coord.item(0)),
                                        int(pose_3d_world_coord.item(1)),
                                        int(pose_3d_world_coord.item(2))]])
            else:
                print('{obj_ID:5} {obj_cls:15} {obj_score:1.3f}'
                      .format(obj_ID=str(int(obj_boxes[i][4])), obj_cls=test_set.object_classes[obj_ind], obj_score=obj_scores[i]))
                node_feature_list.append([test_set.object_classes[obj_ind],
                                        str(int(obj_boxes[i][4])),
                                        [0, 0, 0, 0],
                                        [0,0,0]])

        # Construct Dataframes of node feature
        rows = []
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
                relation_list.append([str(subject_IDs[i]), predicate_inds[i], str(object_IDs[i])])
        
        # Add relation info to DataFrame
        rows.append(relation_list)
        self.data.loc[len(self.data)] = rows
        self.data = self.data.rename(index={idx:"img"+str(idx)})

        print('.........................................................................')
        print(self.data.node_feature)
        '''
        data.node_feature[0] : image number select
        data.node_feature[0][0] : show one pack of object node info
        data.node_feature[0][0][0] : select node's info [0~6] (name, obj_idx, ...) 
        '''
        print('.........................................................................')
        print(self.data.relation) # similar format as node_feature

        for i in range(len(self.data.node_feature[self.img_count])):
            box_color_bgr = colorlist[int(self.data.node_feature[self.img_count][i][1])]
            box_color_rgb = box_color_bgr[::-1]
            box_color_hex = webcolors.rgb_to_hex(box_color_rgb)
            sg.attr('node', style="", color=box_color_hex)
            sg.node('struct'+str(self.data.node_feature[self.img_count][i][1]), '''<
            <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="3">
              <TR>
                <TD PORT="'''+str(self.data.node_feature[self.img_count][i][1])+'''" ROWSPAN="2">'''+str(self.data.node_feature[self.img_count][i][0])+'''</TD>
                <TD COLSPAN="1">ID</TD>
                <TD COLSPAN="3">3D Position</TD>
              </TR>
              <TR>
                <TD>'''+str(self.data.node_feature[self.img_count][i][1])+'''</TD>
                <TD>'''+str(self.data.node_feature[self.img_count][i][3][0]) + '''</TD>
                <TD>'''+str(self.data.node_feature[self.img_count][i][3][1])+'''</TD>
                <TD>'''+str(self.data.node_feature[self.img_count][i][3][2])+'''</TD>
              </TR>
            </TABLE>>''')
            #sg.node_attr.update(style='filled', color = webcolors.rgb_to_hex(colorlist[int(self.data.node_feature[self.img_count][i][1])]))
        for j in range(len(self.data.relation[self.img_count])):
            sg.edge('struct'+str(self.data.relation[self.img_count][j][0])+':'+str(self.data.relation[self.img_count][j][0]),
                    'struct'+str(self.data.relation[self.img_count][j][2])+':'+str(self.data.relation[self.img_count][j][2]),
                    str(test_set.predicate_classes[self.data.relation[self.img_count][j][1]]))
        #sg.render('scene_graph_test/scene-graph_'+str(idx)+'.gv.pdf', view=True, cleanup=False)
        #sg.clear()

        #sg.view()
        self.img_count+=1

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
