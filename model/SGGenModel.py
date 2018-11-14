import sys
sys.path.append('./FactorizableNet')
import models as models
import numpy as np
import torch.nn.functional as F
import torch


VG_DR_NET_PRED_IGNORES=(0,6,10,18,19,20,22,23,24)
VG_DR_NET_OBJ_IGNORES = (0,1,2,3,4,5,9,12,14,15,16,17,18,19,20,21,22,23,24,25,27,29,34,37,38,39,43,44,46,47,48,51,64,65,\
            67,68,71,72,73,77,78,79,83,94,97,103,111,112,113,114,115,120,121,126,127,128,131,138,151,152,154,156,162,163,172,175,177,\
            178,180,182,183,184,200,209,210,212,215,219,221,223,224,228,230,232,234,235,237,239,243,244,246,276,277,278,\
            286,290,291,292,293,296,301,304,307,314,319,\
            320,325,331,334,335,337,340,341,350,354,358,360,361,366,372,373,380,384,396,399,93,124,338,370,379 )
'''
objects_ignored (VG-DR-NET)
0 __background__
1 field            2 zebra            3 sky               4 track           5 train          9 background
12 sheep           14 grass           15 baby             16 ear            17 leg           18 eye
19 tail            20 head            21 nose             22 skateboarder   23 arm           24 foot
25 skateboard      27 hand            29 man              34 hydrant        37 sidewalk      38 curb
39 road            43 people          44 car              46 bus            47 tire          48 lady
49 letter          50 leaf            51 boy              64 cloud          65 kite          66 pants
67 beach           68 woman           71 dog              72 building       73 frisbee       77 hair
78 face            79 shorts          83 cat              94 tennisplayer   97 girl          101 elephant
103 bird           111 shadow         112 sleeve          113 tenniscourt   114 surface      115 finger
120 snow           121 sunglasses     126 skier           128 player        131 wrist        138 horse
151 hill           152 fence          154 cow             156 bear          162 river        163 railing
172 kid            175 mouth          177 air             178 distance      180 feet         182 wave
183 guy            184 reflection     200 stem            209 paw           210 branch       212 forest
215 rail           219 whiskers       221 neck            223 necklace      224 duck         228 giraffe
230 mane           232 beard          234 sun             235 shore         237 tower        239 gravel
243 awning         244 tent           246 teeth           276 pitcher       277 uniform      278 body
286 animal         290 child          291 licenseplate    292 catcher       293 umpire       296 batter
301 bridge         304 park           307 runway          314 theoutdoors   319 van          320 beak           
325 shoulder       331 surfer         334 bluesky         335 whiteclouds   337 vehicle      340 baseball       
341 baseballplayer 350 hoof           354 snowboarder     358 shade         360 spectator    361 knee           
366 platform       372 weeds          373 treetrunk       396 ripples       399 tusk
93 tennisracket    124 skipole        338 streetsign      370 t-shirt       379 tennisball
'''

VG_MSDN_PRED_IGNORES=(9,12, 18,20,22, 27, 28, 30, 31, 32, 35, 48)
VG_MSDN_OBJ_IGNORES=(0,2,11,18,20,22,24,25,26,27,29,30,31,34,35, 37, 39, 40, 41, 43, 44, 45, 49,50,52,54, 55,\
                     56, 58, 60, 67, 68, 72, 78, 79, 80, 81, 83, 84, 85, 88, 90, 92, 93, 95, 96, 97, 100, 103, 104,\
                     107, 113, 115, 118, 119, 121, 127, 128, 129, 130, 133, 135, 136, 142, 143, 145, 147, 150)
'''
objects_ignored (VG-MSDN)
0 __background__   2 kite             11 sky              18 hill           20 woman         22 animal
24 bear            25 wave            26 giraffe          27 background     29 foot          30 shadow
31 lady            34 sand            35 nose             37 sidewalk       39 fence         43 hair
44 street          45 zebra           49 girl             50 arm            52 leaf          54 dirt
55 boat            56 bird            58 leg              60 surfer         67 boy           68 cow            
72 road            78 cloud           79 sheep            80 horse          81 eye           83 neck          
84 tail            85 vehicle         88 head             90 bus            92 train         93 child         
95 ear             96 reflection      97 car              100 cat           103 grass        104 toilet       
107 ocean          113 snow           115 field           118 branch        119 elephant     121 beach        
127 mountain       128 track          129 hand            130 plane         133 skier        135 man          
136 building       142 dog            143 face            145 person        147 truck        150 wing
'''


class SGGen_DR_NET(models.HDN_v2.factorizable_network_v4s.Factorizable_network):
    def __init__(self,args, trainset, opts,
                 ):
        super(SGGen_DR_NET,self).__init__(trainset,opts)
        print(args.path_opt.split('/')[-1])
        if args.path_opt.split('/')[-1].strip() == 'VG-DR-Net.yaml' and args.dataset =='scannet':
            predicates_ignored = VG_DR_NET_PRED_IGNORES
            self.predicates_mask = torch.ByteTensor(40000,25).fill_(0).cuda()
            self.predicates_mask[:,predicates_ignored] = 1
            objects_ignored=VG_DR_NET_OBJ_IGNORES
            self.objects_mask = torch.ByteTensor(200,400).fill_(0)
            self.objects_mask[:,objects_ignored] = 1
            self.objects_mask = self.objects_mask.cuda()

        else:
            self.predicates_mask = torch.ByteTensor(40000, 25).fill_(0).cuda()
            self.objects_mask = torch.ByteTensor(200, 400).fill_(0).cuda()

        self.object_class_filter =[]
        self.predicate_class_filter =[]

    def forward_eval(self, im_data, im_info, gt_objects=None):
        # Currently, RPN support batch but not for MSDN
        features, object_rois = self.rpn(im_data, im_info)
        if gt_objects is not None:
            gt_rois = np.concatenate([np.zeros((gt_objects.shape[0], 1)),
                                      gt_objects[:, :4],
                                      np.ones((gt_objects.shape[0], 1))], 1)
        else:
            gt_rois = None
        object_rois, region_rois, mat_object, mat_phrase, mat_region = self.graph_construction(object_rois, gt_rois=gt_rois)
        # roi pool
        pooled_object_features = self.roi_pool_object(features, object_rois).view(len(object_rois), -1)
        pooled_object_features = self.fc_obj(pooled_object_features)
        pooled_region_features = self.roi_pool_region(features, region_rois)
        pooled_region_features = self.fc_region(pooled_region_features)
        bbox_object = self.bbox_obj(F.relu(pooled_object_features))

        for i, mps in enumerate(self.mps_list):
            pooled_object_features, pooled_region_features = \
                mps(pooled_object_features, pooled_region_features, mat_object, mat_region, object_rois, region_rois)

        pooled_phrase_features = self.phrase_inference(pooled_object_features, pooled_region_features, mat_phrase)
        pooled_object_features = F.relu(pooled_object_features)
        pooled_phrase_features = F.relu(pooled_phrase_features)

        cls_score_object = self.score_obj(pooled_object_features)
        cls_score_object.data.masked_fill_(self.objects_mask, -float('inf'))
        cls_prob_object = F.softmax(cls_score_object, dim=1)
        #print(cls_score_object,cls_score_object.shape)
        cls_score_predicate = self.score_pred(pooled_phrase_features)
        cls_score_predicate.data.masked_fill_(self.predicates_mask, -float('inf'))
        cls_prob_predicate = F.softmax(cls_score_predicate, dim=1)
        #print(cls_prob_predicate, cls_prob_predicate.shape)

        if self.learnable_nms:
            selected_prob, _ = cls_prob_object[:, 1:].max(dim=1, keepdim=False)
            reranked_score = self.nms(pooled_object_features, selected_prob, object_rois)
        else:
            reranked_score = None


        return (cls_prob_object, bbox_object, object_rois, reranked_score), \
                (cls_prob_predicate, mat_phrase, region_rois.size(0)),


class SGGen_MSDN(models.HDN_v2.factorizable_network_v4.Factorizable_network):
    def __init__(self,args, trainset, opts,
                 ):
        super(SGGen_MSDN,self).__init__(trainset,opts)
        print(args.path_opt.split('/')[-1])
        if args.path_opt.split('/')[-1].strip() == 'VG-MSDN.yaml' and args.dataset =='scannet':
            predicates_ignored = VG_MSDN_PRED_IGNORES
            self.predicates_mask = torch.ByteTensor(40000,51).fill_(0).cuda()
            self.predicates_mask[:,predicates_ignored] = 1
            objects_ignored=VG_MSDN_OBJ_IGNORES
            self.objects_mask = torch.ByteTensor(200,151).fill_(0)
            self.objects_mask[:,objects_ignored] = 1
            self.objects_mask = self.objects_mask.cuda()

        else:
            self.predicates_mask = torch.ByteTensor(40000, 51).fill_(0).cuda()
            self.objects_mask = torch.ByteTensor(200, 151).fill_(0).cuda()

    def forward_eval(self, im_data, im_info, gt_objects=None):

        # Currently, RPN support batch but not for MSDN
        features, object_rois = self.rpn(im_data, im_info)
        if gt_objects is not None:
            gt_rois = np.concatenate([np.zeros((gt_objects.shape[0], 1)),
                                      gt_objects[:, :4],
                                      np.ones((gt_objects.shape[0], 1))], 1)
        else:
            gt_rois = None
        object_rois, region_rois, mat_object, mat_phrase, mat_region = self.graph_construction(object_rois, gt_rois=gt_rois)
        # roi pool
        pooled_object_features = self.roi_pool_object(features, object_rois).view(len(object_rois), -1)
        pooled_object_features = self.fc_obj(pooled_object_features)
        pooled_region_features = self.roi_pool_region(features, region_rois)
        pooled_region_features = self.fc_region(pooled_region_features)
        bbox_object = self.bbox_obj(F.relu(pooled_object_features))

        for i, mps in enumerate(self.mps_list):
            pooled_object_features, pooled_region_features = \
                mps(pooled_object_features, pooled_region_features, mat_object, mat_region)

        pooled_phrase_features = self.phrase_inference(pooled_object_features, pooled_region_features, mat_phrase)
        pooled_object_features = F.relu(pooled_object_features)
        pooled_phrase_features = F.relu(pooled_phrase_features)

        cls_score_object = self.score_obj(pooled_object_features)
        cls_score_object.data.masked_fill_(self.objects_mask, -float('inf'))
        cls_prob_object = F.softmax(cls_score_object, dim=1)
        cls_score_predicate = self.score_pred(pooled_phrase_features)
        cls_score_predicate.data.masked_fill_(self.predicates_mask, -float('inf'))
        cls_prob_predicate = F.softmax(cls_score_predicate, dim=1)

        if self.learnable_nms:
            selected_prob, _ = cls_prob_object[:, 1:].max(dim=1, keepdim=False)
            reranked_score = self.nms(pooled_object_features, selected_prob, object_rois)
        else:
            reranked_score = None


        return (cls_prob_object, bbox_object, object_rois, reranked_score), \
                (cls_prob_predicate, mat_phrase, region_rois.size(0)),
