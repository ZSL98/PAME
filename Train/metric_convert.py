from typing import OrderedDict
import cv2
import os
import sys
import csv
import time
import logging
import numpy as np
import torch
import random
import argparse
from typing import Any, Dict, List, Optional, Union
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn


################### Import packages for resnet ####################
###################################################################
# resnet_s1 includes post_layers while partial_resnet does not
# resnet_s1 has dual-heads
import torchvision.models as models
from Mytrain.networks import resnet_s1, partial_resnet
import torchvision.datasets as torchvision_datasets


################### Import packages for openseg ###################
###################################################################
# SpatialOCRNet_with_exit has dual-heads, while SpatialOCRNet_with_only_exit has only one exit
from openseg.lib.models.nets.ocrnet_with_exit import SpatialOCRNet_with_only_exit, SpatialOCRNet_with_exit
from openseg.lib.utils.tools.configer import Configer
from openseg.lib.datasets.data_loader import DataLoader
from openseg.lib.metrics import running_score as rslib
from openseg.segmentor.tools.data_helper import DataHelper
from openseg.segmentor.tools.evaluator import get_evaluator
from openseg.segmentor.etrainer import ETrainer
from openseg import main


################### Import packages for posenet ###################
###################################################################
# PoseResNetwthExit has dual-heads
from pose_estimation.lib.models.pose_resnet import PoseResNetwthExit, PoseResNetwthOnlyExit
from pose_estimation.lib.core.config import update_config, config, get_model_name
from pose_estimation.lib.core.function import validate_exit
from pose_estimation.lib.models import pose_resnet
from pose_estimation.lib.utils.utils import create_logger
from pose_estimation.lib.core.loss import DistillationBasedLoss
from pose_estimation.lib import dataset
from pose_estimation.lib.utils.transforms import flip_back
from pose_estimation.lib.core.evaluate import accuracy
from pose_estimation.lib.core.inference import get_final_preds
from pose_estimation.lib.utils.vis import save_debug_images


################### Import packages for Bert ######################
###################################################################
from bert_train.modeling_bert import BertWithExit, BertWithExit_s1, BertWithExit_s2, BertWithSinglehead
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.utils import logging as transformers_logging
from transformers.trainer_utils import get_last_checkpoint
import transformers
import datasets
from dataclasses import dataclass, field
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


################### Import packages for Wav2Vec2 ##################
###################################################################
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
import re
import soundfile as sf
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, TrainingArguments, Trainer
from Wav2Vec2.wav2vec2_model import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Config, Wav2Vec2Encoder, Wav2Vec2EncoderLayer
from Wav2Vec2.wav2vec2_model import Wav2Vec2RawEncoder, Wav2Vec2_with_exit

logger = logging.getLogger(__name__)

# For openseg and posenet, the parameters of the dual-head-model 
# are transfered from the ee-head-model and the final-head model. \
# Then I realized there is no need to do the load_state_dict, \
# so I simply use net_wth_eehead and net_wth_finalhead to convert metrics.

class convert_resnet:
        
    def load_resnet(split_point):
        net_wth_finalhead = models.resnet101(pretrained=True)
        net_wth_eehead_dict = torch.load("/home/slzhang/projects/ETBA/Train/Mytrain/models/checkpoint.pth.tar."+str(split_point))
        net_wth_eehead = partial_resnet(start_point=split_point, end_point=split_point, simple_exit=False)

        dict_new = OrderedDict()
        for k,v in net_wth_eehead.state_dict().items():
            dict_new[k] = net_wth_eehead_dict['state_dict']['module.'+k]

        net_wth_eehead.load_state_dict(dict_new)

        return net_wth_eehead, net_wth_finalhead

    def eval_resnet(net_wth_eehead, net_wth_finalhead, p_thres):
        valdir = '/home/slzhang/projects/Shallow-Deep-Networks-backup/data/imagenet/ILSVRC/Data/CLS-LOC/val'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        val_loader = torch.utils.data.DataLoader(
            torchvision_datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=512, shuffle=False,
            num_workers=4, pin_memory=True)

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        top1_f = AverageMeter('Acc_final@1', ':6.2f')
        top5_f = AverageMeter('Acc_final@5', ':6.2f')
        pass_acc1 = AverageMeter('Acc1@Pass', ':6.2f')
        pass_ratio1 = AverageMeter('Ratio@Pass', ':6.2f')
        moveon_acc1 = AverageMeter('Acc1@Moveon', ':6.2f')
        moveon_ratio1 = AverageMeter('Ratio@Moveon', ':6.2f')
        avg_acc1 = AverageMeter('Acc1@Avg', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, pass_acc1, moveon_acc1, moveon_ratio1, avg_acc1],
            prefix='Test: ')

        # switch to evaluate mode
        net_wth_eehead.eval()
        net_wth_finalhead.eval()
        net_wth_eehead = torch.nn.DataParallel(net_wth_eehead).cuda()
        net_wth_finalhead = torch.nn.DataParallel(net_wth_finalhead).cuda()

        criterion = nn.CrossEntropyLoss().cuda()

        with torch.no_grad():
            end = time.time()
            moveon_dict = dict()
            for i, (images, target) in enumerate(val_loader):

                # compute output
                images = images.cuda()
                target = target.cuda()

                exit_output = net_wth_eehead(images)
                output = net_wth_finalhead(images)

                loss = criterion(exit_output, target)

                # measure accuracy and record loss
                matrices = convert_resnet.validate_resnet(p_thres, exit_output, output, target, topk=(1, 5))
                acc1 = matrices[0]
                acc5 = matrices[2]
                acc1_final = matrices[1]
                acc5_final = matrices[3]
                p_acc = matrices[4]
                p_ratio = matrices[5]
                m_acc = matrices[6]
                m_ratio = matrices[7]
                moveon_dict[i] = matrices[8]

                p_acc = p_acc.cuda()
                p_ratio = p_ratio.cuda()
                m_acc = m_acc.cuda()
                m_ratio = m_ratio.cuda()

                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                top1_f.update(acc1_final[0], images.size(0))
                top5_f.update(acc5_final[0], images.size(0))
                pass_acc1.update(p_acc, images.size(0))
                pass_ratio1.update(p_ratio, images.size(0))
                moveon_acc1.update(m_acc, images.size(0))
                moveon_ratio1.update(m_ratio, images.size(0))
                avg_acc1.update(p_acc*p_ratio+m_acc*m_ratio, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    # wandb.log({"acc1": acc1[0], "acc5": acc5[0], "pass_acc": p_acc, "pass_ratio": p_ratio})
                    progress.display(i)

            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))

        return pass_acc1.avg.cpu().item(), moveon_acc1.avg.cpu().item(), moveon_ratio1.avg.cpu().item(), avg_acc1.avg.cpu().item()

    def validate_resnet(p_thres, output, final_output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            confidence = p_thres
            m = nn.Softmax(dim=1)
            softmax_output = m(output)
            softmax_final_output = m(final_output)

            pass_indicator = torch.max(softmax_output, 1)[0] > confidence
            moveon_indicator = ~pass_indicator
            pass_cnt = sum(pass_indicator)
            moveon_cnt = sum(moveon_indicator)
            correct_indicator = torch.max(softmax_output, 1)[1] == target
            # final_correct_indicator = torch.max(softmax_final_output + softmax_output, 1)[1] == target
            final_correct_indicator = torch.max(softmax_final_output, 1)[1] == target
            pass_correct_indicator = pass_indicator & correct_indicator
            moveon_correct_indicator = moveon_indicator & final_correct_indicator
            pass_correct_cnt = sum(pass_correct_indicator)
            moveon_correct_cnt = sum(moveon_correct_indicator)
            # print(str(int(pass_correct_cnt)) + '/' + str(int(pass_cnt)))
            if pass_cnt != 0:
                pass_acc = pass_correct_cnt.float().mul_(100.0 / pass_cnt)
            else:
                pass_acc = torch.tensor(0.0)

            if moveon_cnt != 0:
                moveon_acc = moveon_correct_cnt.float().mul_(100.0 / moveon_cnt)
            else:
                moveon_acc = torch.tensor(0.0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            _, pred_f = final_output.topk(maxk, 1, True, True)
            pred_f = pred_f.t()
            correct_f = pred_f.eq(target.view(1, -1).expand_as(pred_f))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
                correct_f_k = correct_f[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_f_k.mul_(100.0 / batch_size))            

            res.append(pass_acc)
            res.append(pass_cnt/batch_size)
            res.append(moveon_acc)
            res.append(moveon_cnt/batch_size)
            res.append(moveon_indicator)
            return res

class convert_posenet:

    def load_posenet(split_point):
        net_wth_finalhead = torch.load("/home/slzhang/projects/ETBA/Train/pose_estimation/models/pytorch/pose_mpii/pose_resnet_101_384x384.pth.tar")
        net_wth_eehead = torch.load("/home/slzhang/projects/ETBA/Train/pose_estimation/output/mpii/pose_resnet_101/384x384_d256x3_adam_lr1e-3/checkpoint.pth.tar."+str(split_point))

        # for k,v in net_wth_eehead['state_dict'].items():
        #     print(k)

        cfg = "/home/slzhang/projects/ETBA/Train/pose_estimation/experiments/mpii/resnet101/384x384_d256x3_adam_lr1e-3.yaml"
        update_config(cfg)

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = config.CUDNN.ENABLED

        net_wth_dualheads = eval(config.MODEL.NAME+'.get_pose_net_with_exit')(
            config, is_train=True, start_point = split_point
        )

        dict_finalhead = net_wth_finalhead.copy()
        dict_finalhead_keys = list(net_wth_finalhead.keys())
        dict_eehead = net_wth_eehead['state_dict'].copy()
        dict_dualhead = OrderedDict()

        i = 0
        for k,v in net_wth_dualheads.state_dict().items():
            if 'num_batches_tracked' not in k and 'deconv' not in k and 'final' not in k and 'exit' not in k:
                dict_dualhead[k] = dict_finalhead[dict_finalhead_keys[i]]
                i = i + 1

        for k,v in net_wth_dualheads.state_dict().items():
            if 'num_batches_tracked' not in k:
                if 'exit' in k:
                    dict_dualhead[k] = dict_eehead['module.'+k]
                elif 'head' in k:
                    dict_dualhead[k] = dict_eehead['module.'+k[5:]]
                elif 'backbone_s1' not in k:
                    dict_dualhead[k] = dict_finalhead[k]

        net_wth_dualheads.load_state_dict(dict_dualhead)
        return net_wth_dualheads

    def eval_posenet(net, p_thres, n_thres):
        net = net.cuda()
        net.eval()

        logger, final_output_dir, tb_log_dir = create_logger(
        config, "/home/slzhang/projects/ETBA/Train/pose_estimation/experiments/mpii/resnet101/384x384_d256x3_adam_lr1e-3.yaml", 'train')

        gpus = [int(i) for i in config.GPUS.split(',')]
        net = torch.nn.DataParallel(net, device_ids=gpus).cuda()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
            config,
            config.DATASET.ROOT,
            config.DATASET.TEST_SET,
            False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.TEST.BATCH_SIZE*len(gpus),
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True
        )

        criterion = DistillationBasedLoss(C=0.5, maxprob = 0.5,
            use_target_weight=config.LOSS.USE_TARGET_WEIGHT
        ).cuda()

        moveon_ratio, metric_avg = convert_posenet.validate_posenet(p_thres, n_thres, config, valid_loader, valid_dataset, net,
                                    criterion, final_output_dir)

        return moveon_ratio, metric_avg

    def validate_posenet(p_thres, n_thres, 
                        config, val_loader, val_dataset, model, criterion, output_dir, writer_dict=None):
        batch_time = AverageMeter('batch_time', ':6.2f')
        losses = AverageMeter('losses', ':6.2f')
        acc = AverageMeter('acc', ':6.2f')
        moveon_ratio = AverageMeter('Ratio@Pass', ':6.2f')

        # switch to evaluate mode
        model.eval()

        num_samples = len(val_dataset)
        all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                            dtype=np.float32)
        all_boxes = np.zeros((num_samples, 6))
        image_path = []
        filenames = []
        imgnums = []
        idx = 0
        with torch.no_grad():
            end = time.time()
            for i, (input, target, target_weight, meta) in enumerate(val_loader):
                # compute output
                output, exit_output = model(input)
                if config.TEST.FLIP_TEST:
                    # this part is ugly, because pytorch has not supported negative index
                    # input_flipped = model(input[:, :, :, ::-1])
                    input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                    input_flipped = torch.from_numpy(input_flipped).cuda()
                    output_flipped, exit_output_flipped = model(input_flipped)
                    output_flipped = flip_back(output_flipped.cpu().numpy(),
                                            val_dataset.flip_pairs)
                    output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
                    exit_output_flipped = flip_back(exit_output_flipped.cpu().numpy(),
                                            val_dataset.flip_pairs)
                    exit_output_flipped = torch.from_numpy(exit_output_flipped.copy()).cuda()

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped[:, :, :, 1:] = \
                            output_flipped.clone()[:, :, :, 0:-1]
                        exit_output_flipped[:, :, :, 1:] = \
                            exit_output_flipped.clone()[:, :, :, 0:-1]
                        # exit_output_flipped[:, :, :, 0] = 0

                    output = (output + output_flipped) * 0.5
                    exit_output = (exit_output + exit_output_flipped) * 0.5

                pixel_confidence = p_thres
                num_threshold = n_thres
                for j in range(output.shape[0]):
                    if len(exit_output[j][exit_output[j] > pixel_confidence]) > num_threshold:
                        moveon_ratio.update(0, 1)
                    else:
                        moveon_ratio.update(1, 1)
                        exit_output[j] = output[j]

                # for j in range(output.shape[0]):
                #     if sorted(exit_output[j])[-num_threshold] > pixel_confidence:
                #         moveon_ratio.update(0, 1)
                #     else:
                #         moveon_ratio.update(1, 1)
                #         exit_output[j] = output[j]                        


                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)

                loss = criterion(output, exit_output, target, target_weight)

                num_images = input.size(0)
                # measure accuracy and record loss
                losses.update(loss.item(), num_images)
                _, avg_acc, cnt, pred = accuracy(exit_output.cpu().numpy(),
                                                target.cpu().numpy())

                # prefix = '{}_{}'.format(os.path.join(output_dir, 'test'), 0)
                # save_debug_images_with_exit(config, input, meta, target, pred*4, 
                #                             output, exit_output, prefix)
                # exit()

                acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                c = meta['center'].numpy()
                s = meta['scale'].numpy()
                score = meta['score'].numpy()

                preds, maxvals = get_final_preds(
                    config, exit_output.clone().cpu().numpy(), c, s)

                all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
                all_preds[idx:idx + num_images, :, 2:3] = maxvals
                # double check this all_boxes parts
                all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
                all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
                all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
                all_boxes[idx:idx + num_images, 5] = score
                image_path.extend(meta['image'])
                if config.DATASET.DATASET == 'posetrack':
                    filenames.extend(meta['filename'])
                    imgnums.extend(meta['imgnum'].numpy())

                idx += num_images

                if i % config.PRINT_FREQ == 0:
                    # wandb.log({"acc":acc, "loss":loss})
                    msg = 'Test: [{0}/{1}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                        'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                        'Moveon_Ratio {moveon_ratio.val:.3f} ({moveon_ratio.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time,
                            loss=losses, acc=acc, moveon_ratio=moveon_ratio)
                    logger.info(msg)

                    # prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                    # save_debug_images(config, input, meta, target, pred*4, exit_output,
                    #                   prefix)

            name_values, perf_indicator = val_dataset.evaluate(
                config, all_preds, output_dir, all_boxes, image_path,
                filenames, imgnums)

            _, full_arch_name = get_model_name(config)
            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value, full_arch_name)
            else:
                _print_name_value(name_values, full_arch_name)

        return moveon_ratio.avg, acc.avg


class convert_openseg:

    def load_openseg(split_point):
        # load the backbone of the network with dual-heads

        net_wth_finalhead = torch.load("/home/slzhang/projects/ETBA/Train/openseg/checkpoints/cityscapes/ocrnet_resnet101_s33_latest.pth")
        # for k,v in net_wth_finalhead['state_dict'].items():
        #     print(k)

        config = Configer(configs="/home/slzhang/projects/ETBA/Train/openseg/configs/cityscapes/R_101_D_8_with_exit.json")
        config.update(["network", "split_point"], split_point)
        net_wth_eehead = torch.load("/home/slzhang/projects/ETBA/Train/openseg/checkpoints/cityscapes/ocrnet_resnet101_s"+str(split_point)+"_latest.pth")
        # for k,v in net_wth_eehead['state_dict'].items():
        #     print(k)


        net_wth_dualheads = SpatialOCRNet_with_exit(config)
        # for k,v in net_wth_dualheads.state_dict().items():
        #     print(k)

        dict_finalhead = net_wth_finalhead['state_dict'].copy()
        dict_eehead = net_wth_eehead['state_dict'].copy()
        dict_dualhead = OrderedDict()

        for k,v in net_wth_dualheads.state_dict().items():
            if 'num_batches_tracked' not in k:
                if 'backbone_s1.resinit' in k:
                    dict_dualhead[k] = dict_finalhead['module.backbone'+k[11:]]
                elif 'backbone_s1.pre' in k:
                    dict_dualhead[k] = dict_finalhead['module.backbone.'+k[16:]]
                elif 'backbone_s1.exit' in k:
                    dict_dualhead[k] = dict_eehead['module.'+k]
                elif 'backbone_s2.layer3' in k:
                    new_idx = int(k.split('.')[2]) + config.get("network", "split_point")-7
                    dict_dualhead[k] = dict_finalhead['module.backbone.layer3.'+str(new_idx)+'.'+'.'.join(k.split('.')[3:])]
                elif 'backbone_s2.layer4' in k:
                    dict_dualhead[k] = dict_finalhead['module.backbone.'+k[12:]]
                elif 'conv_3x3_s1' in k:
                    dict_dualhead[k] = dict_eehead['module.conv_3x3' + k[11:]]
                elif 'conv_3x3_s2' in k:
                    dict_dualhead[k] = dict_finalhead['module.conv_3x3' + k[11:]]
                elif 'head_s1' in k:
                    dict_dualhead[k] = dict_eehead['module.'+k.split('.')[0][:-3]+'.'+'.'.join(k.split('.')[1:])]
                elif 'head_s2' in k:
                    dict_dualhead[k] = dict_finalhead['module.'+k.split('.')[0][:-3]+'.'+'.'.join(k.split('.')[1:])]     

        net_wth_dualheads.load_state_dict(dict_dualhead)
        return net_wth_dualheads

    def eval_openseg(net, p_thres, n_thres):
        net = net.cuda()
        net.eval()
        config = Configer(configs="/home/slzhang/projects/ETBA/Train/openseg/configs/cityscapes/R_101_D_8_with_exit.json")
        abs_data_dir = "/home/slzhang/projects/ETBA/Train/openseg/data/cityscapes"
        config.update(["data", "data_dir"], abs_data_dir)
        config.add(["gpu"], [0, 1, 2, 3])
        config.add(["network", "gathered"], "n")
        config.add(["network", "resume"], None)
        config.add(["optim", "group_method"], None)
        config.add(["data", "include_val"], False)
        config.add(["data", "include_coarse"], False)
        config.add(["data", "include_atr"], False)
        config.add(["data", "only_coarse"], False)
        config.add(["data", "only_mapillary"], False)
        config.add(["data", "drop_last"], True)
        config.add(["network", "loss_balance"], False)
        data_loader = DataLoader(config)
        etrainer = ETrainer(config)
        data_helper = DataHelper(config, etrainer)
        val_loader = data_loader.get_valloader()
        evaluator = get_evaluator(config, etrainer)

        moveon_ratio = AverageMeter('Ratio@Pass', ':6.2f')
        mIoU = AverageMeter('mIoU@Avg', ':6.2f')
        mIoU_s1 = AverageMeter('mIoU_s1@Avg', ':6.2f')
        mIoU_s2 = AverageMeter('mIoU_s2@Avg', ':6.2f')

        for j, data_dict in enumerate(val_loader):
            (inputs, targets), batch_size = data_helper.prepare_data(data_dict)
            with torch.no_grad():
                outputs = net(*inputs)
                if isinstance(outputs, torch.Tensor):
                    outputs = [outputs]
                metas = data_dict["meta"]
                output_s1 = outputs[1].permute(0, 2, 3, 1)[0].cpu().numpy()
                output_s2 = outputs[3].permute(0, 2, 3, 1)[0].cpu().numpy()
                final_output = convert_openseg.validate_openseg(p_thres, n_thres, output_s1, output_s2, moveon_ratio)

                labelmap = np.argmax(final_output, axis=-1)
                ori_target = metas[0]['ori_target']
                RunningScore = rslib.RunningScore(config)
                RunningScore.update(labelmap[None], ori_target[None])
                rs = RunningScore.get_mean_iou()
                if moveon_ratio.val == 0:
                    mIoU_s1.update(rs)
                else:
                    mIoU_s2.update(rs)
                mIoU.update(rs)
                os.system("clear")
                print("mIoU: {}".format(mIoU.avg))
                print("mIoU_s1: {}".format(mIoU_s1.avg))
                print("mIoU_s2: {}".format(mIoU_s2.avg))
                print("moveon_ratio: {}".format(moveon_ratio.avg))

        return mIoU_s1.avg, mIoU_s2.avg, moveon_ratio.avg, mIoU.avg

    def validate_openseg(p_thres, n_thres, 
                        output_s1, output_s2, moveon_ratio):
        # Shape of output_s1/output_s2: (1024, 2048, 19)
        pixel_threshold = p_thres
        num_threshold = n_thres
        pixel_confidence = np.amax(output_s1, axis=-1)
        # print(pixel_confidence)
        pixel_over_threshold = pixel_confidence > pixel_threshold
        num_pixel_over_threshold = pixel_over_threshold.sum()
        # print(num_pixel_over_threshold)
        if num_pixel_over_threshold > num_threshold:
            moveon_ratio.update(0, 1)
            return output_s1
        else:
            moveon_ratio.update(1, 1)
            return output_s2

class convert_bert:
    def __init__(self, split_point, task_name) -> None:
        super().__init__()
        self.task_name = task_name
        self.split_point = split_point
        self.raw_datasets = load_dataset("glue", task_name, cache_dir=None)
        self.is_regression = task_name == "stsb"
        if not self.is_regression:
            self.label_list = self.raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        self.p_thres = 0.8

    def load_bert(self):

        net_wth_finalhead_dict = torch.load("/home/slzhang/projects/ETBA/Train/bert_train/models/"+self.task_name+'/exit12/pytorch_model.bin')
        net_wth_eehead_dict = torch.load("/home/slzhang/projects/ETBA/Train/bert_train/models/"+self.task_name+'/exit'+str(self.split_point)+'/pytorch_model.bin')
        
        config = AutoConfig.from_pretrained(
            'bert-base-cased',
            num_labels=self.num_labels,
            finetuning_task=self.task_name,
        )
        net_wth_eehead = BertWithSinglehead.from_pretrained('bert-base-cased', config=config)
        net_wth_eehead.add_exit(self.split_point)
        net_wth_finalhead = BertWithSinglehead.from_pretrained('bert-base-cased', config=config)
        net_wth_finalhead.add_exit(12)

        dict_eehead = OrderedDict()
        dict_finalhead = OrderedDict()

        for k,v in net_wth_eehead.state_dict().items():
            dict_eehead[k] = net_wth_eehead_dict[k]
        net_wth_eehead.load_state_dict(dict_eehead)

        for k,v in net_wth_finalhead.state_dict().items():
            dict_finalhead[k] = net_wth_finalhead_dict[k]
        net_wth_finalhead.load_state_dict(dict_finalhead)

        eval_eehead = self.bert_aux(net_wth_eehead)
        eval_finalhead = self.bert_aux(net_wth_finalhead)

        return eval_eehead, eval_finalhead

    def bert_aux(self, model):
        # See all possible arguments in src/transformers/training_args.py
        # or by passing the --help flag to this script.
        # We now keep distinct sets of args, for a cleaner separation of concerns.
        
        # TODO: rewrite the parser
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Set seed before initializing model.
        set_seed(training_args.seed)

        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=self.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # Preprocessing the raw_datasets
        if self.task_name is not None:
            sentence1_key, sentence2_key = task_to_keys[self.task_name]
        else:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            non_label_column_names = [name for name in self.raw_datasets["train"].column_names if name != "label"]
            if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None

        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        if (
            model.config.label2id != PretrainedConfig(num_labels=self.num_labels).label2id
            and self.task_name is not None
            and not self.is_regression
        ):
            # Some have all caps in their config, some don't.
            label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
            if list(sorted(label_name_to_id.keys())) == list(sorted(self.label_list)):
                label_to_id = {i: int(label_name_to_id[self.label_list[i]]) for i in range(self.num_labels)}
            else:
                logger.warning(
                    "Your model seems to have been trained with labels, but they don't match the dataset: ",
                    f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(self.label_list))}."
                    "\nIgnoring the model labels as a result.",
                )
        elif self.task_name is None and not self.is_regression:
            label_to_id = {v: i for i, v in enumerate(self.label_list)}

        if label_to_id is not None:
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        elif self.task_name is not None and not self.is_regression:
            model.config.label2id = {l: i for i, l in enumerate(self.label_list)}
            model.config.id2label = {id: label for label, id in config.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
            return result

        with training_args.main_process_first(desc="dataset map pre-processing"):
            self.raw_datasets = self.raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if "validation" not in self.raw_datasets and "validation_matched" not in self.raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        self.eval_dataset = self.raw_datasets["validation_matched" if self.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            self.val_dataset = self.eval_dataset.select(range(data_args.max_eval_samples))

        # Get the metric function
        if self.task_name is not None:
            metric = load_metric("glue", self.task_name)
        else:
            metric = load_metric("accuracy")

        # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
        # predictions and label_ids field) and has to return a dictionary string to float.
        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
            if self.task_name is not None:
                result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result
            elif self.is_regression:
                return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
            else:
                return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None

        # Initialize our Trainer
        eval_model = Trainer(
            model=model,
            args=training_args,
            train_dataset=None,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        return eval_model

    def eval_bert(self, eval_eehead, eval_finalhead):

        passAcc = AverageMeter('passAcc@Avg', ':6.2f')
        moveonAcc = AverageMeter('mvonAcc@Avg', ':6.2f')
        moveonRatio = AverageMeter('mvonRatio@Avg', ':6.2f')

        tasks = [self.task_name]
        eval_datasets = [self.eval_dataset]
        if self.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(self.raw_datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            predictions = eval_eehead.predict(eval_dataset, metric_key_prefix="predict").predictions
            final_predictions = eval_finalhead.predict(eval_dataset, metric_key_prefix="predict").predictions

            pass_acc, moveon_acc, moveon_cnt = self.validate_bert(predictions, final_predictions, eval_dataset)

            passAcc.update(pass_acc, len(eval_dataset))
            moveonAcc.update(moveon_acc, len(eval_dataset))
            moveonRatio.update(moveon_cnt/len(eval_dataset), len(eval_dataset))

        # os.system("clear")
        print("passAcc: {}".format(passAcc.avg))
        print("moveonAcc: {}".format(moveonAcc.avg))
        print("moveon_ratio: {}".format(moveonRatio.avg))
        acc_avg = passAcc.avg * (1-moveonRatio.avg) + moveonAcc.avg * moveonRatio.avg

        return passAcc.avg.item(), moveonAcc.avg.item(), moveonRatio.avg.item(), acc_avg.item()


    def validate_bert(self, output, final_output, eval_dataset):

        output = torch.from_numpy(output)
        final_output = torch.from_numpy(final_output)
        target = torch.tensor(eval_dataset["label"])
        m = nn.Softmax(dim=1)
        softmax_output = m(output)
        softmax_final_output = m(final_output)

        pass_indicator = torch.max(softmax_output, 1)[0] > self.p_thres
        moveon_indicator = ~pass_indicator
        pass_cnt = sum(pass_indicator)
        moveon_cnt = sum(moveon_indicator)
        correct_indicator = torch.max(softmax_output, 1)[1] == target
        final_correct_indicator = torch.max(softmax_final_output, 1)[1] == target
        pass_correct_indicator = pass_indicator & correct_indicator
        moveon_correct_indicator = moveon_indicator & final_correct_indicator
        pass_correct_cnt = sum(pass_correct_indicator)
        moveon_correct_cnt = sum(moveon_correct_indicator)
        # print(str(int(pass_correct_cnt)) + '/' + str(int(pass_cnt)))
        if pass_cnt != 0:
            pass_acc = pass_correct_cnt.float().mul_(100.0 / pass_cnt)
        else:
            pass_acc = torch.tensor(0.0)

        if moveon_cnt != 0:
            moveon_acc = moveon_correct_cnt.float().mul_(100.0 / moveon_cnt)
        else:
            moveon_acc = torch.tensor(0.0)

        return pass_acc, moveon_acc, moveon_cnt

class convert_Wav2Vec2(object):

    def __init__(self, split_point) -> None:
        super().__init__()
        self.split_point = split_point
        self.p_thres = 0.8
        self.tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)


    def remove_special_characters(self, batch):
        batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
        return batch

    def speech_file_to_array_fn(self, batch):
        speech_array, sampling_rate = sf.read(batch["file"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
        return batch

    def prepare_dataset(self, batch):
        # check that all files have the correct sampling rate
        assert (
            len(set(batch["sampling_rate"])) == 1
        ), f"Make sure all inputs have the same sampling rate of {self.processor.feature_extractor.sampling_rate}."

        batch["input_values"] = self.processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
        
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["target_text"]).input_ids
        return batch

    @dataclass
    class DataCollatorCTCWithPadding:
        """
        Data collator that will dynamically pad the inputs received.
        Args:
            processor (:class:`~transformers.Wav2Vec2Processor`)
                The processor used for proccessing the data.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
                among:
                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                different lengths).
            max_length (:obj:`int`, `optional`):
                Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
            max_length_labels (:obj:`int`, `optional`):
                Maximum length of the ``labels`` returned list and optionally padding length (see above).
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
                7.5 (Volta).
        """

        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            batch["labels"] = labels

            return batch

    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer_metric = load_metric("wer")
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


    def load_Wav2Vec2(self, split_point):
        net_wth_finalhead_dict = torch.load('/home/slzhang/projects/ETBA/Train/Wav2Vec2/checkpoints/timit_exit_12/checkpoint-435/pytorch_model.bin')
        net_wth_eehead_dict = torch.load('/home/slzhang/projects/ETBA/Train/Wav2Vec2/checkpoints/timit_exit_{}/checkpoint-435/pytorch_model.bin'.format(split_point))


        timit = load_dataset("timit_asr")
        timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])
        timit = timit.map(self.remove_special_characters)
        timit = timit.map(self.speech_file_to_array_fn, remove_columns=timit.column_names["train"], num_proc=4)
        timit_prepared = timit.map(self.prepare_dataset, remove_columns=timit.column_names["train"], batch_size=8, num_proc=4, batched=True)

        data_collator = self.DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        net_wth_finalhead = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-base", 
            gradient_checkpointing=True, 
            ctc_loss_reduction="mean", 
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        net_wth_eehead = Wav2Vec2_with_exit.from_pretrained(
            "facebook/wav2vec2-base", 
            gradient_checkpointing=True, 
            ctc_loss_reduction="mean", 
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        net_wth_eehead.add_exit(self.split_point)

        dict_eehead = OrderedDict()
        dict_finalhead = OrderedDict()

        for k,v in net_wth_eehead.state_dict().items():
            dict_eehead[k] = net_wth_eehead_dict[k]
        net_wth_eehead.load_state_dict(dict_eehead)

        for k,v in net_wth_finalhead.state_dict().items():
            dict_finalhead[k] = net_wth_finalhead_dict[k]
        net_wth_finalhead.load_state_dict(dict_finalhead)

        eval_eehead = self.bert_aux(net_wth_eehead)
        eval_finalhead = self.bert_aux(net_wth_finalhead)

        return eval_eehead, eval_finalhead

    def eval_Wav2Vec2(self, eval_eehead, eval_finalhead):
        pass

        

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    split_point: int = field(
        metadata={"help": "Split point"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def grid_search(task_name, split_point):
    if task_name == 'imagenet':
        net_wth_eehead, net_wth_finalhead = convert_resnet.load_resnet(split_point)
        for p_thres in np.arange(0, 1.1, 0.1):
            metric_eehead, metric_finalhead, moveon_ratio, metric_avg = convert_resnet.eval_resnet(net_wth_eehead, net_wth_finalhead, p_thres)
            with open('/home/slzhang/projects/ETBA/Train/conversion_results/imagenet_results_{}.csv'.format(split_point), 'a+') as f:
                writer = csv.writer(f)
                writer.writerow([p_thres, metric_eehead, metric_finalhead, moveon_ratio, metric_avg])

    elif task_name == 'posenet':
        net = convert_posenet.load_posenet(split_point)
        for p_thres in np.arange(0.6, 0.84, 0.02):
            for n_thres in [5, 10, 15, 20, 30, 50]:
                moveon_ratio, metric_avg = convert_posenet.eval_posenet(net, p_thres, n_thres)
                with open('/home/slzhang/projects/ETBA/Train/conversion_results/posenet_results_{}.csv'.format(split_point), 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow([p_thres, n_thres, moveon_ratio, metric_avg])

    elif task_name == 'openseg':
        net = convert_openseg.load_posenet(split_point)
        for p_thres in np.arange(1, 10, 1):
            for n_thres in [100000, 200000, 500000, 1000000]:
                metric_eehead, metric_finalhead, moveon_ratio, metric_avg = convert_resnet.eval_resnet(net, p_thres, n_thres)
                with open('/home/slzhang/projects/ETBA/Train/conversion_results/openseg_results_{}.csv'.format(split_point), 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow([p_thres, n_thres, metric_eehead, metric_finalhead, moveon_ratio, metric_avg])

    elif task_name == 'mrpc':
        inst = convert_bert(split_point=split_point, task_name=task_name)
        eval_eehead, eval_finalhead = inst.load_bert()
        for p_thres in np.arange(0, 1.1, 0.1):
            inst.p_thres = p_thres
            metric_eehead, metric_finalhead, moveon_ratio, metric_avg = inst.eval_bert(eval_eehead, eval_finalhead)
            with open('/home/slzhang/projects/ETBA/Train/conversion_results/mrpc_results_{}.csv'.format(split_point), 'a+') as f:
                writer = csv.writer(f)
                writer.writerow([p_thres, metric_eehead, metric_finalhead, moveon_ratio, metric_avg])

    elif task_name == 'Wav2Vec2':
        pass

if __name__ == '__main__':
    task = 'imagenet'
    mode = 'tes'

    if mode == 'test':
        if task == 'posenet':
            net = convert_posenet.load_posenet(20)
            convert_posenet.eval_posenet(net, 0.8, 50)
        elif task == 'openseg':
            net = convert_openseg.load_openseg(20)
            convert_openseg.eval_openseg(net, 8, 0)
        elif task == 'bert':
            inst = convert_bert(split_point=8, task_name='mrpc')
            eval_eehead, eval_finalhead = inst.load_bert()
            inst.eval_bert(eval_eehead, eval_finalhead)
        elif task == 'Wav2Vec2':
            inst = convert_Wav2Vec2(split_point=5)
            inst.load_Wav2Vec2()
    # TODO: For bert, run "python metric_convert.py --output_dir /home/slzhang/projects/ETBA/Train/bert_train/models/tmp --split_point 5 --model_name_or_path bert-base-cased --task_name mrpc --do_eval

    grid_search(task, 14)



# load_backbone()

# load state dict from the stage one model

# load_head()
