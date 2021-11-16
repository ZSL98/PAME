import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.nn import functional as F
import wandb
from main import DistillationBasedLoss, DistillationBasedLoss_only_exit
from networks import partial_resnet, exit_resnet_s1, exit_resnet_s2, exit_resnet_s3, exit_resnet_s4, resnet_s1

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/home/slzhang/projects/Shallow-Deep-Networks-backup/data/imagenet/ILSVRC/Data/CLS-LOC',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
# parser.add_argument('--start-point', default=33, type=int, metavar='N',
#                     help='split the network from the start point')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--split_point', default=None, type=int,
                    help='Split point.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0
features_in_hook = []
features_out_hook = []
# outputs_lists = {}

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global features_in_hook
    global features_out_hook
    args.gpu = gpu

    wandb.init(
        # Set entity to specify your username or team name
        # ex: entity="carey",
        # Set the project where this run will be logged
        project="resnet_train", 
        name="split_point:"+str(args.split_point),
        # Track hyperparameters and run metadata
        config={
        "architecture": "resnet101",
        "dataset": "imagenet",})

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model

    model = partial_resnet(start_point=args.split_point, end_point=args.split_point, simple_exit=False)
    # model_pretrained = models.resnet101(pretrained=True, progress=True)
    model_pretrained = models.resnet101(pretrained=True)
    new_list = list (model.state_dict().keys())
    trained_list = list (model_pretrained.state_dict().keys())

    dict_trained = model_pretrained.state_dict().copy()
    dict_new = model.state_dict().copy()

    i = 0
    for k,v in model.state_dict().items():
        if 'exit' not in k and 'fc' not in k:
            dict_new[new_list[i]] = dict_trained[trained_list[i]]
        i = i + 1
    
    model.load_state_dict(dict_new)

    i = 0
    for k,v in model.named_parameters():
        if 'exit' not in k and 'fc' not in k:
            v.requires_grad=False
        else:
            v.requires_grad=True
        i = i + 1

    # for i in range(626):
    #     dict_new[new_list[i]] = dict_trained[trained_list[i]]

    # model.load_state_dict(dict_new)

    # i = 0
    # for k,v in model.named_parameters():
    #     if i < 314:
    #         v.requires_grad=False
    #     else:
    #         v.requires_grad=True
    #     i = i + 1

    # torch.save(model.state_dict(), "/home/slzhang/projects/ETBA/Train/Mytrain/resnet101.model")
    # model.load_state_dict(torch.load("/home/slzhang/projects/ETBA/Train/Mytrain/resnet101.model"), strict=False)
    # print(len(model.state_dict().keys()))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        print("Case 1")
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        print("Case 2")
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print("Case 3")
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
        # ee = torch.nn.DataParallel(ee).cuda()
        # model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # criterion = DistillationBasedLoss_only_exit(C=0.5, maxprob = 0.5, global_scale = 2.0 * 5/2, n_exits = 2, acc_tops = [1, 5])
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    # filter(lambda p: p.requires_grad, model.parameters())

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # if args.evaluate:
    #     validate(val_loader, model, criterion, args)
    #     return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            print('Save checkpoint!')
            save_checkpoint({
                'epoch': epoch,
                'arch': "resnet101",
                'state_dict':model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def hook(module, fea_in, fea_out):
    global features_in_hook
    global features_out_hook
    # global outputs_lists
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    # outputs_lists[fea_in[0].device].append(fea_out)
    return None

def train(train_loader, model, criterion, optimizer, epoch, args):
    global features_in_hook
    global features_out_hook
    # global outputs_lists
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    pass_acc1 = AverageMeter('Acc@Pass', ':6.2f')
    pass_ratio1 = AverageMeter('Ratio@Pass', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, pass_acc1, pass_ratio1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    # backbone.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        
        # compute output
        images = images.cuda()

        # output, exit_output = model(images)
        # loss = criterion(output, exit_output, target)
        exit_output = model(images)
        loss = criterion(exit_output, target)

        # measure accuracy and record loss
        acc1, acc5, p_acc, p_ratio = accuracy(exit_output, target, topk=(1, 5))
        p_acc = p_acc.cuda()
        p_ratio = p_ratio.cuda()
        losses.update(loss.item(), exit_output.size(0))
        top1.update(acc1[0], exit_output.size(0))
        top5.update(acc5[0], exit_output.size(0))
        pass_acc1.update(p_acc, exit_output.size(0))
        pass_ratio1.update(p_ratio, exit_output.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            wandb.log({"acc1": acc1[0], "acc5": acc5[0], "pass_acc": p_acc, "pass_ratio": p_ratio})
            progress.display(i)

def validate(val_loader, model, criterion, args):
    global features_in_hook
    global features_out_hook
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
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, top1_f, top5_f, pass_acc1, pass_ratio1, moveon_acc1, moveon_ratio1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            images = images.cuda()
            exit_output = model(images)

            loss = criterion(exit_output, target)

            # measure accuracy and record loss
            matrices = val_accuracy(exit_output, target, topk=(1, 5))
            acc1 = matrices[0]
            acc5 = matrices[2]
            acc1_final = matrices[1]
            acc5_final = matrices[3]
            p_acc = matrices[4]
            p_ratio = matrices[5]
            m_acc = matrices[6]
            m_ratio = matrices[7]

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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                # wandb.log({"acc1": acc1[0], "acc5": acc5[0], "pass_acc": p_acc, "pass_ratio": p_ratio})
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        confidence = 0.5
        m = nn.Softmax(dim=1)
        softmax_output = m(output)

        pass_indicator = torch.max(softmax_output, 1)[0] > confidence
        pass_cnt = sum(pass_indicator)
        correct_indicator = torch.max(softmax_output, 1)[1] == target
        pass_correct_indicator = pass_indicator & correct_indicator
        pass_correct_cnt = sum(pass_correct_indicator)
        # print(str(int(pass_correct_cnt)) + '/' + str(int(pass_cnt)))
        if pass_cnt != 0:
            pass_acc = pass_correct_cnt.float().mul_(100.0 / pass_cnt)
        else:
            pass_acc = torch.tensor(0.0)
        # print(pass_acc)


        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        res.append(pass_acc)
        res.append(pass_cnt/batch_size)
        return res

#TODO: The val_accuracy is for multi-output, thus using this will cause error in saving models.
def val_accuracy(output, final_output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        confidence = 0.8
        m = nn.Softmax(dim=1)
        softmax_output = m(output)
        softmax_final_output = m(final_output)

        pass_indicator = torch.max(softmax_output, 1)[0] > confidence
        moveon_indicator = ~pass_indicator
        pass_cnt = sum(pass_indicator)
        moveon_cnt = sum(moveon_indicator)
        correct_indicator = torch.max(softmax_output, 1)[1] == target
        final_correct_indicator = torch.max(softmax_final_output + softmax_output, 1)[1] == target
        # final_correct_indicator = torch.max(softmax_final_output, 1)[1] == target
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
        return res


if __name__ == '__main__':
    main()