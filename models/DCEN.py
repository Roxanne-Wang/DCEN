import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .resnet import *
from  models.MPNCOV import MPNCOV

   
        
class DCEN(nn.Module):
    def __init__(self, args=None):
        super(DCEN, self).__init__()
        ''' param '''
        self.K = args.cmcn_K
        self.m = 0.999
        self.T = 0.07
        self.dim = 256
        num_classes = args.num_classes
        self.att = torch.from_numpy(args.att)
        self.att_dim = self.att.size(1)
        self.att.requires_grad = False
        
        ''' backbone '''
        self.encoder_q = resnet101(num_classes=self.dim,pretrained=True)
        if(args.is_fix):
            for n,p in self.named_parameters():
                if 'fc' not in n:
                    p.requires_grad=False
        feat_dim = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(), self.encoder_q.fc)
        
        
        
        ''' ins dis'''
        self.encoder_k = resnet101(num_classes=self.dim)
        self.encoder_k.fc = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        ''' sem inv '''
        # contastive loss
        self.si_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.si_cls = nn.Sequential(
            nn.Linear(self.att_dim,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,feat_dim),
            nn.LeakyReLU(),
        )
        # predictive loss
        self.si_proj_v = nn.Sequential(
            nn.Linear(feat_dim,1024),
            nn.LeakyReLU(),
        )
        self.si_proj_s = nn.Sequential(
            nn.Linear(feat_dim,1024),
            nn.LeakyReLU(),
        )
        
        
        self.si_pred = nn.Sequential(
                nn.Linear(feat_dim,1024),
                nn.LeakyReLU(),
                nn.Linear(1024,self.att_dim),
                nn.ReLU(),
            )
            
        
        self.si_aux = nn.Linear(feat_dim, num_classes)
        
       
        
        ''' Domain Detection Module '''
        self.ood_proj =  nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1, stride=1, padding=0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.ood_classifier = nn.Linear(int(256*(256+1)/2), num_classes)
        
        
        self.ood_spatial =  nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
                nn.ReLU(inplace=True),   
                nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0,bias=False),
                nn.Sigmoid(),        
            )
        self.ood_channel =  nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(256, int(256/16), kernel_size=1, stride=1, padding=0,bias=False),
                nn.ReLU(inplace=True),   
                nn.Conv2d(int(256/16), 256, kernel_size=1, stride=1, padding=0,bias=False),
                nn.Sigmoid(),        
            )
        
        ''' params ini '''
        for n, m in self.named_modules():
            if 'si' in n or 'ood' in n:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        
    def forward_eval(self, im_q):
        ''' backbone feat '''
        q, feat = self.encoder_q(im_q)  # queries: NxC
      
        
        ''' domain detection part'''
        x = self.ood_proj(feat)
        
        att1 = self.ood_spatial(x)
        att2 = self.ood_channel(x)
        x1 = att2*x+x
        x1 = x1.view(x1.size(0),x1.size(1),-1)
        x2 = att1*x+x
        x2 = x2.view(x2.size(0),x2.size(1),-1)
        # covariance pooling
        x1 = x1 - torch.mean(x1,dim=2,keepdim=True)
        x2 = x2 - torch.mean(x2,dim=2,keepdim=True)
        A = 1./x1.size(2)*x1.bmm(x2.transpose(1,2))
            
        # norm
        x = MPNCOV.SqrtmLayer(A, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        # cls
        logit_od = self.ood_classifier(x)
        
        x = self.si_proj(feat).view(feat.size(0),-1)
        si_cls = self.si_cls(self.att.cuda())
        w_norm = F.normalize(si_cls, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        logit_si = x_norm.mm(w_norm.permute(1,0))
        
        return (logit_od,logit_si),(feat) #aux: classfier of train set (150) .....  si: logit with atributes (200)
        
    def forward_train(self, im_q, im_k, att):
        ''' backbone feat '''
        q, feat = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)
        
        ''' domain detection part '''
        x = self.ood_proj(feat)
        
        att1 = self.ood_spatial(x)
        att2 = self.ood_channel(x)
        x1 = att2*x+x
        x1 = x1.view(x1.size(0),x1.size(1),-1)
        x2 = att1*x+x
        x2 = x2.view(x2.size(0),x2.size(1),-1)
        # covariance pooling
        x1 = x1 - torch.mean(x1,dim=2,keepdim=True)
        x2 = x2 - torch.mean(x2,dim=2,keepdim=True)
        A = 1./x1.size(2)*x1.bmm(x2.transpose(1,2))
            
        
        
        # norm
        x = MPNCOV.SqrtmLayer(A, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        # cls
        logit_od = self.ood_classifier(x)
        
        ''' ins dis '''
        # key img
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        # logits: Nx(1+K)
        logit_id = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logit_id /= self.T
        
        ''' sem inv '''
        x = self.si_proj(feat).view(feat.size(0),-1)
        si_cls = self.si_cls(att)
        w_norm = F.normalize(si_cls, p=2, dim=1) # [N,C]
        x_norm = F.normalize(x, p=2, dim=1) # [N,C]
        logit_si_pos = (w_norm*x_norm).sum(dim=1) # [N]
        logit_aux = self.si_aux(x)
        # predictive
        vis = self.si_proj_v(x) # [N,2048] -> [N,1024]
        sem = self.si_proj_s(si_cls) # [N,2048] -> [N,1024]
        att_pred = self.si_pred(torch.cat([vis,sem],dim=1)) # [N,2048] -> [N,312]
        # neg l
        si_cls = self.si_cls(self.att.cuda()) # [A,C]
        w_norm = F.normalize(si_cls, p=2, dim=1)
        logit_si_neg = x_norm.mm(w_norm.permute(1,0)) # [N,CLS]
        
        return (logit_id,logit_si_pos,logit_si_neg,logit_aux,logit_od),(feat,att_pred)
            
    def forward(self, im_q, im_k=None, att=None, mode='train'):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if mode=='eval':
            logits, feats = self.forward_eval(im_q)
        else:
            logits, feats = self.forward_train(im_q,im_k,att)
            
        return logits,feats
		
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
		
        self.cls_loss = nn.CrossEntropyLoss()#reduce=False)
        self.mse_loss = nn.MSELoss()
        self.Lsi_base = args.Lsi_base
        self.att = torch.from_numpy(args.att)
        
    def forward(self, labels, logits, feats):
        logit_id = logits[0]
        logit_si_pos = logits[1]
        logit_si_neg = logits[2]
        logit_aux = logits[3]
        logit_od = logits[4]
        
        att_pred = feats[1]
        
        ''' od '''
        L_od =  self.cls_loss(logit_od,labels)
        
        ''' ins dis '''
        # labels: positive key indicators
        id_labels = torch.zeros(logit_id.size(0), dtype=torch.long).cuda()
        L_id = self.cls_loss(logit_id,id_labels) * 0.1
        
        ''' sem inv '''
        if self.Lsi_base == True: ###baseline Lsi: without negative loss
            L_si = (1-logit_si_pos).mean()            
        else:
            [N,L] = logit_si_neg.size()
            logit_si_neg = logit_si_pos.view(N,1)-logit_si_neg # [N,CLS]
            idx = torch.arange(N).long()
            logit_si_neg[idx,labels] = 10 # [N,CLS]
            logit_si_neg,_ = logit_si_neg.min(dim=1) # [N]
            L_si = (1-logit_si_pos-torch.clamp(logit_si_neg,-10,0)).mean()
            
            
        L_aux = self.cls_loss(logit_aux,labels)
        
        ''' sem pred '''
        att_target = self.att[labels,:].cuda()
        L_sp = self.mse_loss(att_pred,att_target)
        
        return L_id,L_si,L_sp,L_aux,L_od
		
def dcen(args=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DCEN(args)
    loss_model = Loss(args)
    
    return model,loss_model
	
	
