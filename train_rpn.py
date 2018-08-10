import os
import torch
import numpy as np
import time
from faster_rcnn import network
from faster_rcnn.RPN import RPN # Hierarchical_Descriptive_Model
from faster_rcnn.utils.timer import Timer
from faster_rcnn.utils.HDN_utils import check_recall
from faster_rcnn.datasets.visual_genome_loader import visual_genome
from faster_rcnn.fast_rcnn.config import cfg
import argparse
import pdb
'''Training command example'''
# Train from scratch
# CUDA_VISIBLE_DEVICES=0 python train_rpn.py --max_epoch=10 --step_size=2 --dataset_option=normal --model_name=RPN_full_region_resnet101
# Resume training
# CUDA_VISIBLE_DEVICES=0 python train_rpn.py --max_epoch=10 --step_size=2 --dataset_option=normal --model_name=RPN_full_region_resnet101 --resume_training --resume_model ./output/RPN/RPN_full_region_resnet101_best.h5

parser = argparse.ArgumentParser('Options for training RPN in pytorch')

## training settings
parser.add_argument('--lr', type=float, default=0.01, help='To disable the Lanuage Model ')
parser.add_argument('--max_epoch', type=int, default=5, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='percentage of past parameters to store')
parser.add_argument('--log_interval', type=int, default=500, help='Interval for Logging')
parser.add_argument('--disable_clip_gradient', action='store_true', help='Whether to clip the gradient')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')
parser.add_argument('--step_size', type=int, default=2, help='step to decay the learning rate')
parser.add_argument('--train_blocks', type= int, default = 0, help = 'the number of blocks to train in cnn' )

## Environment Settings
parser.add_argument('--cnn_type', type=str, default ='resnet', help='vgg, resnet, senet')
parser.add_argument('--pretrained_model', type=str, default='model/pretrained_models/VGG_imagenet.npy', help='Path for the to-evaluate model')
parser.add_argument('--dataset_option', type=str, default='small', help='The dataset to use (small | normal | fat)')
parser.add_argument('--output_dir', type=str, default='./output/RPN', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='RPN_region', help='model name for snapshot')
parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
parser.add_argument('--resume_model', type=str, default='./output/trained_model/RPN_region_full_best.h5', help='The model we resume')
args = parser.parse_args()


def train(train_loader, target_net, optimizer, epoch):
    batch_time = network.AverageMeter()
    data_time = network.AverageMeter()
    train_loss = network.AverageMeter()
    train_loss_obj_box = network.AverageMeter()
    train_loss_obj_entropy = network.AverageMeter()
    train_loss_reg_box = network.AverageMeter()
    train_loss_reg_entropy = network.AverageMeter()

    target_net.train()
    end = time.time()
    for i, (im_data, im_info, gt_objects, gt_relationships, gt_regions) in enumerate(train_loader):
        # measure the data loading time
        data_time.update(time.time() - end)

        # Forward pass
        target_net(im_data, im_info.numpy(), gt_objects.numpy()[0], gt_regions.numpy()[0])
        # record loss
        loss = target_net.loss
        # total loss
        train_loss.update(loss.data[0], im_data.size(0))
        # object bbox reg
        train_loss_obj_box.update(target_net.loss_box.data[0], im_data.size(0))
        # object score
        train_loss_obj_entropy.update(target_net.cross_entropy.data[0], im_data.size(0))
        # region bbox reg
        train_loss_reg_box.update(target_net.loss_box_region.data[0], im_data.size(0))
        # region score
        train_loss_reg_entropy.update(target_net.cross_entropy_region.data[0], im_data.size(0))

        # backward
        optimizer.zero_grad()
        loss.backward()
        if not args.disable_clip_gradient:
            network.clip_gradient(target_net, 10.)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if  (i + 1) % args.log_interval == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch_Time: {batch_time.avg:.3f}s\t'
                  'lr: {lr: f}\t'
                  'Loss: {loss.avg:.4f}\n'
                  '\t[object]: '
                  'cls_loss: {cls_loss_object.avg:.3f}\t'
                  'reg_loss: {reg_loss_object.avg:.3f}\n'
                  '\t[region]: '
                  'cls_loss: {cls_loss_region.avg:.3f}\t'
                  'reg_loss: {reg_loss_region.avg:.3f}\t'.format(
                   epoch, i + 1, len(train_loader), batch_time=batch_time,lr=args.lr, 
                   data_time=data_time, loss=train_loss, 
                   cls_loss_object=train_loss_obj_entropy, reg_loss_object=train_loss_obj_box, 
                   cls_loss_region=train_loss_reg_entropy, reg_loss_region=train_loss_reg_box))

def test(test_loader, target_net):
    box_num = np.array([0, 0])
    correct_cnt, total_cnt = np.array([0, 0]), np.array([0, 0])
    print('========== Testing =======')
    target_net.eval()

    batch_time = network.AverageMeter()
    end = time.time()
    for i, (im_data, im_info, gt_objects, gt_relationships, gt_regions) in enumerate(test_loader):
        correct_cnt_t, total_cnt_t = np.array([0, 0]), np.array([0, 0])
        # Forward pass
        object_rois, region_rois = target_net(im_data, im_info.numpy(), gt_objects.numpy(), gt_regions.numpy())[1:]
        box_num[0] += object_rois.size(0)
        box_num[1] += region_rois.size(0)
        correct_cnt_t[0], total_cnt_t[0] = check_recall(object_rois, gt_objects[0].numpy(), 50)
        correct_cnt_t[1], total_cnt_t[1] = check_recall(region_rois, gt_regions[0].numpy(), 50)
        correct_cnt += correct_cnt_t
        total_cnt += total_cnt_t
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 100 == 0 and i > 0:
            print('[{0}/{10}]  Time: {1:2.3f}s/img).'
                  '\t[object] Avg: {2:2.2f} Boxes/im, Top-50 recall: {3:2.3f} ({4:d}/{5:d})' 
                  '\t[region] Avg: {6:2.2f} Boxes/im, Top-50 recall: {7:2.3f} ({8:d}/{9:d})'.format(
                    i + 1, batch_time.avg, 
                    box_num[0] / float(i + 1), correct_cnt[0] / float(total_cnt[0])* 100, correct_cnt[0], total_cnt[0], 
                    box_num[1] / float(i + 1), correct_cnt[1] / float(total_cnt[1])* 100, correct_cnt[1], total_cnt[1], 
                    len(test_loader)))

    recall = correct_cnt / total_cnt.astype(np.float)
    print('====== Done Testing ====')
    return recall

if __name__ == '__main__':
    print("Loading training set and testing set...")
    train_set = visual_genome(args.dataset_option, 'train')
    test_set = visual_genome('small', 'test')
    print("Done.")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    net = RPN(not args.use_normal_anchors, cnn_type=args.cnn_type)
    if args.cnn_type == 'vgg':
        optimizer = torch.optim.SGD(list(net.parameters())[26:], lr=args.lr, momentum=args.momentum,
                                    weight_decay=0.0005)
    elif args.cnn_type in ['resnet','senet']:
        params = list(net.conv1.parameters()) +\
                 list(net.score_conv.parameters()) +\
                 list(net.bbox_conv.parameters()) +\
                 list(net.conv1_region.parameters()) +\
                 list(net.score_conv_region.parameters()) +\
                 list(net.bbox_conv_region.parameters())
        if args.train_blocks > 0:
            for i in range(args.train_blocks):
                params += list(net.features[-i].parameters())
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=0.0005)
    else:
        raise NotImplementedError

    if args.resume_training:
        print('Resume training from: {}'.format(args.resume_model))
        if len(args.resume_model) == 0:
            raise Exception('[resume_model] not specified')
        net, optimizer, start_epoch, best_recall, recall \
            = network.load_checkpoint(args.resume_model,net,optimizer,method='h5')

        # network.load_net(args.resume_model, net)
        # checkpoint = torch.load(args.resume_model[:-2] + 'pth',
        #                         map_location=lambda storage, loc: storage)
        # # net.load_state_dict(checkpoint['model'])
        # start_epoch = checkpoint['epoch'] + 1
        # optim_dict = checkpoint['optimizer']
        # best_recall = checkpoint['best_recall']
        # optimizer.load_state_dict(optim_dict)
        # for state in optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.cuda()
        # net.cuda()

        '''
        # Testing
        recall = test(test_loader, net)

        print('Epoch[{epoch:d}]: '
              'Recall: '
              'object: {recall[0]: .3f}%% (Best: {best_recall[0]: .3f}%%)'
              'region: {recall[1]: .3f}%% (Best: {best_recall[1]: .3f}%%)'.format(
                epoch = checkpoint['epoch'], recall=recall * 100, best_recall=best_recall * 100))
        '''
    else:
        start_epoch = 0
        best_recall = np.array([0.0, 0.0])
        net.cuda()
        print('Training from scratch...Initializing network...')

    # network.set_trainable(net.features, requires_grad=False)
    if not os.path.exists('./output'):
        os.mkdir('./output')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for epoch in range(start_epoch, args.max_epoch):

        # Training
        train(train_loader, net, optimizer, epoch)

        # Testing
        recall = test(test_loader, net)
        print('Epoch[{epoch:d}]: '
              'Recall: '
              'object: {recall[0]: .3f}%% (Best: {best_recall[0]: .3f}%%)'
              'region: {recall[1]: .3f}%% (Best: {best_recall[1]: .3f}%%)'.format(
            epoch=epoch, recall=recall * 100, best_recall=best_recall * 100))
        # update learning rate
        if epoch % args.step_size == 0:
            args.disable_clip_gradient = True
            args.lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        # snapshot the state
        network.save_checkpoint(args, net, optimizer, best_recall, recall, epoch)


        # save_name = os.path.join(args.output_dir, '{}_epoch_{}.h5'.format(args.model_name, epoch))
        # network.save_net(save_name, net)
        # network.save_checkpoint({
        #     'epoch': epoch,
        #     'model': net.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'best_recall': best_recall,
        #     'recall': recall,
        # }, save_name[:-2] + 'pth')
        # if np.all(recall > best_recall):
        #     best_recall = recall
        #     save_name = os.path.join(args.output_dir, '{}_best.h5'.format(args.model_name, epoch))
        #     network.save_net(save_name, net)
        #     network.save_checkpoint({
        #         'epoch': epoch,
        #         'model': net.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, save_name[:-2] + 'pth')
        # print('save model: {}'.format(save_name))
