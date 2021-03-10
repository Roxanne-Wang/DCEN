#from __future__ import print_function
import argparse
import os
import random
import shutil
import time
import warnings
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import sys
import torchvision.transforms as transforms
import datasets
import models
from utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--imgpath', '-im', metavar='IMG', default='',
                    help='image path')  
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--save_path', '-s', metavar='SAVE', default='',
                    help='saving path')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--is_fix', dest='is_fix', action='store_true',
                    help='is_fix.')
parser.add_argument('--train_bn_infix', dest='train_bn_infix', action='store_true',
                    help='if true: train bn in fix phase.')
parser.add_argument('--load_model_part', default='all', type=str, metavar='PATH',
                    help='load whether part of the model from fix model')           
                    
''' data proc '''
parser.add_argument('--val_aug', default='v1', type=str, help='')
parser.add_argument('--aug', default='v1', type=str, help='')
parser.add_argument('--sem_aug', default='None', type=str, help='')

''' opt '''
parser.add_argument('--lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--epoch_decay', default=30, type=int,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--opt', default='sgd', type=str, help='')
''' model '''
parser.add_argument('--cmcn_K', default=2800, type=int,
                    help='')

''' loss '''
parser.add_argument('--w_Lid', default=0.1, type=float,
                    help='')
parser.add_argument('--w_Lsp', default=0, type=float,
                    help='')
parser.add_argument('--Lsi_base', default=False, action='store_true',
                    help='Lsi, if set True: train without negative loss. if set False: train with negative')                
                               
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    ''' save path '''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ''' random seed '''
    if args.seed is not None:
        random.seed(args.seed)
    else:
        args.seed = random.randint(1, 10000)
        
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    print('==> random seed:',args.seed)
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    ''' data load info '''
    args.data  = 'awa2'
    img_path = args.imgpath
    
    
    data_info = h5py.File(os.path.join('./data',args.data,'data_info.h5'), 'r')
    nc = data_info['all_att'][...].shape[0]
    sf_size = data_info['all_att'][...].shape[1]
    semantic_data = {'seen_class':data_info['seen_class'][...],
                     'unseen_class': data_info['unseen_class'][...],
                     'all_class':np.arange(nc),
                     'all_att': data_info['all_att'][...]}
    # load semantic data
    args.num_classes = nc
    args.att = semantic_data['all_att']
    
    ''' model building '''
    model,criterion = models.__dict__[args.arch](args=args)
   # model,criterion = eval('models.'+args.arch)(args=args)
    model = torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda(args.gpu)

    
    ''' optimizer '''
    odr_params = [v for k, v in model.named_parameters() if 'ood' in k]
    zsr_params = [v for k, v in model.named_parameters() if 'ood' not in k]
    
    od_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, odr_params),
                         args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    zs_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, zsr_params), 0.001,
                                    betas=(0.5,0.999),weight_decay=args.weight_decay)
                                    
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                         args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    

    ''' optionally resume from a checkpoint'''
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #args.start_epoch = checkpoint['epoch']
            if(best_prec1==0):
                best_prec1 = checkpoint['best_prec1']
            print('=> pretrained acc {:.4F}'.format(best_prec1))
            model_dict = model.state_dict()
            pretrained_dict = checkpoint['state_dict']
            if args.load_model_part == "only_si":
                
                if "dvbe" not in args.resume:                    
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and ("si" in k or "ood" in k)}#"queue" not in k}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join('./data',args.data,'train.list')
    valdir1 = os.path.join('./data',args.data,'test_seen.list')
    valdir2 = os.path.join('./data',args.data,'test_unseen.list')

    train_transforms,val_transforms,sem_transforms = preprocess_strategy(args)


    train_dataset = datasets.ImageFolder(
            img_path, traindir, TwoCropsTransform(train_transforms),
            semantic_transform=sem_transforms,att=args.att)
            
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None,drop_last=True)

    val_loader1 = torch.utils.data.DataLoader(
        datasets.ImageFolder(img_path,valdir1, val_transforms,att=args.att),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    val_loader2 = torch.utils.data.DataLoader(
        datasets.ImageFolder(img_path,valdir2, val_transforms,att=args.att),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        adjust_learning_rate(od_optimizer, zs_optimizer, optimizer, epoch)
            
           # # evaluate on validation set
         #   prec1 = validate(val_loader1, val_loader2, semantic_data, model, criterion)  
            # train for one epoch
        train(train_loader, model, criterion, optimizer, od_optimizer,zs_optimizer, epoch,is_fix=args.is_fix, args=args)
        
        
        # evaluate on validation set
        prec1 = validate(val_loader1, val_loader2, semantic_data, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save model
        if args.is_fix:
            save_path = os.path.join(args.save_path,'fix.model')
        else:
            save_path = os.path.join(args.save_path,args.arch+('_{:.4f}.model').format(best_prec1))
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                #'optimizer' : optimizer.state_dict(),
            },filename=save_path)
            print('saving!!!!')
        

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()


def train(train_loader, model, criterion, optimizer, od_optimizer,zs_optimizer, epoch,is_fix,args):    
    # switch to train mode
    model.train()
    if(is_fix and not args.train_bn_infix):
        freeze_bn(model) 
    

    end = time.time()
    for i, (input, target, att) in enumerate(train_loader):
        # measure data loading time
        if args.gpu is not None:
            input[0] = input[0].cuda(args.gpu, non_blocking=True)
            input[1] = input[1].cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        att = att.cuda(args.gpu, non_blocking=True)

        # compute output
        logits,feats = model(im_q=input[0], im_k=input[1], att=att)
        L_id,L_si,L_sp,L_aux, L_od = criterion(target,logits,feats)
        
        loss_id = args.w_Lid*L_id
        loss_si = L_si + L_aux + args.w_Lsp*L_sp
        loss_od = L_od

        # compute gradient and do SGD step
        if is_fix:
            od_optimizer.zero_grad()
            loss_od.backward()
            od_optimizer.step()
        
            zs_optimizer.zero_grad()
            (loss_id+loss_si).backward()
            zs_optimizer.step()
        else:
            optimizer.zero_grad()
            (loss_id+loss_si+loss_od).backward()
            optimizer.step()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] loss:'.format
                   (epoch, i, len(train_loader)),end='')
            print('L_id {:.4f} L_si {:.4f} L_sp {:.4f}'.format(L_id.item(),L_si.item(),L_sp.item()))

def validate(val_loader1, val_loader2, semantic_data, model, criterion):

    ''' load semantic data'''
    seen_c = semantic_data['seen_class']
    unseen_c = semantic_data['unseen_class']
    all_c = semantic_data['all_class']
    
    # switch to evaluate mode
    model.eval()
    
    if args.val_aug == "v2":#flipping test
        test_flip = True
    else:
        test_flip = False

    with torch.no_grad():
        end = time.time()
        for i, (input, target,_) in enumerate(val_loader1):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            if test_flip:                
                [N,M,C,H,W] = input.size()
                input = input.view(N*M,C,H,W)   #flipping test
            
            # inference
            logits,feats = model(input,mode='eval')
            
            if test_flip: 
                odr_logit = F.softmax(logits[0],dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
                zsl_logit = F.softmax(logits[1],dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
            else:
                odr_logit = logits[0].cpu().numpy()
                zsl_logit = logits[1].cpu().numpy()
            zsl_logit_s = zsl_logit.copy();zsl_logit_s[:,unseen_c]=-1
            zsl_logit_t = zsl_logit.copy();zsl_logit_t[:,seen_c]=-1
			
            # evaluation
            if(i==0):
                gt_s = target.cpu().numpy()
                odr_pre_s = np.argmax(odr_logit, axis=1)
                if test_flip: 
                    odr_prob_s = odr_logit
                else:
                    odr_prob_s = softmax(odr_logit)
                
                zsl_pre_sA = np.argmax(zsl_logit, axis=1)
                zsl_pre_sS = np.argmax(zsl_logit_s, axis=1)
                if test_flip:
                    zsl_prob_s = zsl_logit_t
                else: 
                    zsl_prob_s = softmax(zsl_logit_t)
            else:
                gt_s = np.hstack([gt_s,target.cpu().numpy()])
                odr_pre_s = np.hstack([odr_pre_s,np.argmax(odr_logit, axis=1)])
                if test_flip:
                    odr_prob_s = np.vstack([odr_prob_s,odr_logit])
                else:
                    odr_prob_s = np.vstack([odr_prob_s,softmax(odr_logit)])
                zsl_pre_sA = np.hstack([zsl_pre_sA,np.argmax(zsl_logit, axis=1)])
                zsl_pre_sS = np.hstack([zsl_pre_sS,np.argmax(zsl_logit_s, axis=1)])
                if test_flip:
                    zsl_prob_s = np.vstack([zsl_prob_s,zsl_logit_t])
                else:
                    zsl_prob_s = np.vstack([zsl_prob_s,softmax(zsl_logit_t)])

        for i, (input, target,_) in enumerate(val_loader2):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            
            if test_flip:                
                [N,M,C,H,W] = input.size()
                input = input.view(N*M,C,H,W)   #flipping test

            # inference
            logits,feats = model(input,mode='eval')
            if test_flip: 
                odr_logit = F.softmax(logits[0],dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
                zsl_logit = F.softmax(logits[1],dim=1).view(N,M,-1).mean(dim=1).cpu().numpy()
            else:
                odr_logit = logits[0].cpu().numpy()
                zsl_logit = logits[1].cpu().numpy()
            zsl_logit_s = zsl_logit.copy();zsl_logit_s[:,unseen_c]=-1
            zsl_logit_t = zsl_logit.copy();zsl_logit_t[:,seen_c]=-1
                
            if(i==0):
                gt_t = target.cpu().numpy()
                odr_pre_t = np.argmax(odr_logit, axis=1)
                if test_flip: 
                    odr_prob_t = odr_logit
                else:
                    odr_prob_t = softmax(odr_logit)
                zsl_pre_tA = np.argmax(zsl_logit, axis=1)
                zsl_pre_tT = np.argmax(zsl_logit_t, axis=1)
                if test_flip: 
                    zsl_prob_t = zsl_logit_t
                else:
                    zsl_prob_t = softmax(zsl_logit_t)
            else:
                gt_t = np.hstack([gt_t,target.cpu().numpy()])
                odr_pre_t = np.hstack([odr_pre_t,np.argmax(odr_logit, axis=1)])
                if test_flip: 
                    odr_prob_t = np.vstack([odr_prob_t,odr_logit])
                else:
                    odr_prob_t = np.vstack([odr_prob_t,softmax(odr_logit)])
                zsl_pre_tA = np.hstack([zsl_pre_tA,np.argmax(zsl_logit, axis=1)])
                zsl_pre_tT = np.hstack([zsl_pre_tT,np.argmax(zsl_logit_t, axis=1)])
                if test_flip: 
                    zsl_prob_t = np.vstack([zsl_prob_t,zsl_logit_t])
                else:
                    zsl_prob_t = np.vstack([zsl_prob_t,softmax(zsl_logit_t)])
                
        odr_prob = np.vstack([odr_prob_s,odr_prob_t])
        zsl_prob = np.vstack([zsl_prob_s,zsl_prob_t])
        gt = np.hstack([gt_s,gt_t])
        
        H = post_process(odr_prob, zsl_prob, gt, gt_s.shape[0], seen_c,unseen_c, args.data)
        print('current H {:.4f}'.format(H))              
    return H


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)        
        
def adjust_learning_rate(od_optimizer, zs_optimizer, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.epoch_decay))
    for param_group in od_optimizer.param_groups:
        param_group['lr'] = lr
        
    lr = 0.001 * (0.1 ** (epoch // args.epoch_decay))
    for param_group in zs_optimizer.param_groups:
        param_group['lr'] = lr
        
    lr = args.lr * (0.1 ** (epoch // args.epoch_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
