import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torch
from PIL import Image
from PIL import ImageFilter
import random
import transforms as att_trans

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                                

def swap(img, crop):
    def crop_image(image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list

    widthcut, highcut = img.size
    img = img.crop((10, 10, widthcut-10, highcut-10))
    images = crop_image(img, crop)
    pro = 5
    if pro >= 5:          
        tmpx = []
        tmpy = []
        count_x = 0
        count_y = 0
        k = 1
        RAN = 2
        for i in range(crop[1] * crop[0]):
            tmpx.append(images[i])
            count_x += 1
            if len(tmpx) >= k:
                tmp = tmpx[count_x - RAN:count_x]
                random.shuffle(tmp)
                tmpx[count_x - RAN:count_x] = tmp
            if count_x == crop[0]:
                tmpy.append(tmpx)
                count_x = 0
                count_y += 1
                tmpx = []
            if len(tmpy) >= k:
                tmp2 = tmpy[count_y - RAN:count_y]
                random.shuffle(tmp2)
                tmpy[count_y - RAN:count_y] = tmp2
        random_im = []
        for line in tmpy:
            random_im.extend(line)
        
        # random.shuffle(images)
        width, high = img.size
        iw = int(width / crop[0])
        ih = int(high / crop[1])
        toImage = Image.new('RGB', (iw * crop[0], ih * crop[1]))
        x = 0
        y = 0
        for i in random_im:
            i = i.resize((iw, ih), Image.ANTIALIAS)
            toImage.paste(i, (x * iw, y * ih))
            x += 1
            if x == crop[0]:
                x = 0
                y += 1
    else:
        toImage = img
    toImage = toImage.resize((widthcut, highcut))
    return toImage

class Randomswap(object):
    def __init__(self, size):
        self.size = size
        self.size = (int(size), int(size))

    def __call__(self, img):
        return swap(img, self.size)


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
        
        
def freeze_bn(model):
    for n, m in model.named_modules():
        if isinstance(m,nn.BatchNorm2d):
            m.eval()
            
def get_RANK(query_semantic, test_mask, classes):
    query_semantic = query_semantic.cpu().numpy()
    test_mask = test_mask.cpu().numpy()
    query_semantic = query_semantic/np.linalg.norm(query_semantic,2,axis=1,keepdims=True)
    test_mask = test_mask/np.linalg.norm(test_mask,2,axis=1,keepdims=True)
    dist = np.dot(query_semantic, test_mask.transpose())
    return classes[np.argmax(dist, axis=1)]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def softmax(x):
    """Compute the softmax of vector x."""
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x,axis=1,keepdims=True)
    return softmax_x 

def compute_domain_accuracy(predict_label, domain):
    num = predict_label.shape[0]
    n = 0
    for i in predict_label:
        if i in domain:
            n +=1
            
    return float(n)/num

def compute_class_accuracy_total( true_label, predict_label, classes):
    nclass = len(classes)
    acc_per_class = np.zeros((nclass, 1))
    for i, class_i in enumerate(classes):
        idx = np.where(true_label == class_i)[0]
        acc_per_class[i] = (sum(true_label[idx] == predict_label[idx])*1.0 / len(idx))
    return np.mean(acc_per_class)

def entropy(probs): 
    """ Computes entropy. """ 
    max_score = np.max(probs,axis=1)   
    return -max_score * np.log(max_score)




def preprocess_strategy(args):
    if args.aug=='v1':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])  
    elif args.aug=='v2':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor(),
            normalize
            ])

    elif args.aug=='v3':
        train_transforms = transforms.Compose([
            transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.5),
            transforms.RandomResizedCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomApply([Randomswap(3)], p=0.2),
            transforms.ToTensor(),
            normalize
            ])
    
    if args.val_aug=='v1':
        val_transforms = transforms.Compose([
            transforms.Resize(480),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ]) 
    elif args.val_aug=='v2':   #flipping testing   
        val_transforms = transforms.Compose([
            transforms.Resize(480),
            transforms.CenterCrop(448),
            transforms.Lambda(lambda x: [x,transforms.RandomHorizontalFlip(p=1.0)(x)]),
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
            transforms.Lambda(lambda crops: [normalize(crop) for crop in crops]),
            transforms.Lambda(lambda crops: torch.stack(crops))
        ])
    

    if args.sem_aug=='None':
        sem_transforms=None        
    elif args.sem_aug=='v2':
        sem_transforms = transforms.Compose([
            transforms.RandomApply([att_trans.AttMask(mask_ratio=0.02)], p=0.2),
            att_trans.AttToTensor(),
        ]) 
    return train_transforms,val_transforms,sem_transforms
    
    
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6




def opt_domain_acc(cls_s,cls_t):
    ''' source domain '''
    opt_acc_s = 0
    num_s = cls_s.shape[0]
    max_score_s = np.max(cls_s,axis=1)   
          
    opt_acc_t = 0
    num_t = cls_t.shape[0]
    max_score_t = np.max(cls_t,axis=1)
    
    max_H = 0
    opt_tau = 0
    for step in range(10):
        tau = 0.1*step
        
        idx = np.where(max_score_s>tau)
        acc_s = float(idx[0].shape[0])/num_s 
        
        idx = np.where(max_score_t<tau)
        acc_t = float(idx[0].shape[0])/num_t
         
        H = 2*acc_s*acc_t/(acc_s+acc_t) 
        if H>max_H:
            opt_acc_t = acc_t
            opt_acc_s = acc_s
            max_H = H
            opt_tau = tau
    return opt_acc_s,opt_acc_t,opt_tau
            
def post_process(v_prob,a_prob,gt, split_num, seen_c,unseen_c,data):
    v_max = np.max(v_prob,axis=1)
    H_v = entropy(v_prob)   
    v_pre = np.argmax(v_prob,axis=1)
    
    a_max = np.max(v_prob,axis=1)
    H_a = entropy(a_prob)
    a_pre = np.argmax(a_prob,axis=1) 
        
    opt_S = 0
    opt_U = 0
    opt_H = 0
    opt_Ds = 0
    opt_Du = 0
    opt_tau = 0
        
    for step in range(9):
        base = 0.02*step+0.83            
        tau = -base* np.log(base)
        pre = v_pre
        for idx,class_i in enumerate(pre):
            if(v_max[idx]-base<0):
                pre[idx] = a_pre[idx]                   
                
        pre_s = pre[:split_num];pre_t = pre[split_num:]
        gt_s = gt[:split_num];gt_t = gt[split_num:]
        S = compute_class_accuracy_total(gt_s, pre_s,seen_c)
        U = compute_class_accuracy_total(gt_t, pre_t,unseen_c)
        Ds = compute_domain_accuracy(pre_s,seen_c)
        Du = compute_domain_accuracy(pre_t,unseen_c)
        H = 2*S*U/(S+U)        
        if H>opt_H:
             opt_H = H            
    return opt_H