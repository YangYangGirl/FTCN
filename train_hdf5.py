import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import sys
import random
from sbi_utils.sbi import SBI_Dataset
from sbi_utils.taylor_hdf5 import Taylor_Video_HDF5_Dataset
from sbi_utils.scheduler import LinearDecayLR
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from sbi_utils.logs import log
from sbi_utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm

from utils.plugin_loader import PluginLoader
from config import config as ftcn_cfg
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
from test_tools.common import detect_all, grab_all_frames
from test_tools.utils import get_crop_box
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
import argparse
from tqdm import tqdm

class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma, milestones, last_epoch = -1):
        self.init_lr = [group['lr'] for group in optimizer.param_groups]
        self.gamma = gamma
        self.milestones = milestones
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        global_step = self.last_epoch #iteration number in pytorch
        gamma_power = ([0] + [i + 1 for i, m in enumerate(self.milestones) if global_step >= m])[-1]
        return [init_lr * (self.gamma ** gamma_power) for init_lr in self.init_lr]


def compute_accuray(pred,true):
    pred_idx=pred.argmax(dim=1).cpu().data.numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)

def main(args):
    cfg=load_json(args.config)
    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    image_size=cfg['image_size']
    batch_size=cfg['batch_size']
    print('image_size = ', image_size)
    train_dataset=Taylor_Video_HDF5_Dataset(phase='train',image_size=image_size,n_frames=32)
    val_dataset=Taylor_Video_HDF5_Dataset(phase='val',image_size=image_size,n_frames=32)
    
    # train_dataset=SBI_Dataset(phase='train',image_size=image_size)
    # val_dataset=SBI_Dataset(phase='val',image_size=image_size)
   

    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size//2,
                        shuffle=True,
                        collate_fn=train_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        drop_last=True,
                        worker_init_fn=train_dataset.worker_init_fn
                        )
    val_loader=torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=val_dataset.collate_fn,
                        num_workers=4,
                        pin_memory=True,
                        worker_init_fn=val_dataset.worker_init_fn
                        )
    
    ftcn_cfg.init_with_yaml()
    ftcn_cfg.update_with_yaml("ftcn_tt.yaml")
    ftcn_cfg.freeze()

    model = PluginLoader.get_classifier(ftcn_cfg.classifier_type)() # 'i3d_temporal_var_fix_dropout_tt_cfg'

    model=model.to('cuda')
    
    iter_loss=[]
    train_losses=[]
    test_losses=[]
    train_accs=[]
    test_accs=[]
    val_accs=[]
    val_losses=[]
    n_epoch=cfg['epoch']
    
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=0.0001)
    scheduler = MultiStepLR(optimizer, gamma=1, milestones=[10, 20])

    # pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=0.0001, momentum=0.9, weight_decay=0.005)
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    # scheduler = MultiStepLR(optimizer, gamma = 1, milestones = [10, 20])
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    last_loss=99999

    now=datetime.now()
    save_path='output/{}_'.format(args.session_name)+now.strftime(os.path.splitext(os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
    os.mkdir(save_path)
    os.mkdir(save_path+'weights/')
    os.mkdir(save_path+'logs/')
    logger = log(path=save_path+"logs/", file="losses.logs")

    criterion=nn.CrossEntropyLoss()

    last_auc=0
    last_val_auc=0
    weight_dict={}
    n_weight=5
    for epoch in range(n_epoch):
        np.random.seed(seed + epoch)
        train_loss=0.
        train_acc=0.
        model.train(mode=True)
        for step,data in enumerate(tqdm(train_loader)):
            img=data['video'].to(device, non_blocking=True).float()
            img = img.permute(0, 2, 1, 3, 4)
            target=data['label'].to(device, non_blocking=True).long()[:, 0]
            output=model.training_step(img, target)
            loss=criterion(output,target)
            loss_value=loss.item()
            iter_loss.append(loss_value)
            train_loss+=loss_value
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            train_acc+=acc
        lr_scheduler.step()
        train_losses.append(train_loss/len(train_loader))
        train_accs.append(train_acc/len(train_loader))

        log_text="Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
                        epoch+1,
                        n_epoch,
                        train_loss/len(train_loader),
                        train_acc/len(train_loader),
                        )

        model.train(mode=False)
        val_loss=0.
        val_acc=0.
        output_dict=[]
        target_dict=[]
        np.random.seed(seed)
        for step,data in enumerate(tqdm(val_loader)):
            img=data['video'].to(device, non_blocking=True).float()
            target=data['label'].to(device, non_blocking=True).long()[:, 0]
            
            with torch.no_grad():
                output=model(img)
                loss=criterion(output,target)
            
            loss_value=loss.item()
            iter_loss.append(loss_value)
            val_loss+=loss_value
            acc=compute_accuray(F.log_softmax(output,dim=1),target)
            val_acc+=acc
            output_dict+=output.softmax(1)[:,1].cpu().data.numpy().tolist()
            target_dict+=target.cpu().data.numpy().tolist()
        val_losses.append(val_loss/len(val_loader))
        val_accs.append(val_acc/len(val_loader))
        val_auc=roc_auc_score(target_dict,output_dict)
        log_text+="val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
                        val_loss/len(val_loader),
                        val_acc/len(val_loader),
                        val_auc
                        )
     

        if len(weight_dict)<n_weight:
            save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
            weight_dict[save_model_path]=val_auc
            torch.save({
                    "model":model.state_dict(),
                    "optimizer":model.optimizer.state_dict(),
                    "epoch":epoch
                },save_model_path)
            last_val_auc=min([weight_dict[k] for k in weight_dict])

        elif val_auc>=last_val_auc:
            save_model_path=os.path.join(save_path+'weights/',"{}_{:.4f}_val.tar".format(epoch+1,val_auc))
            for k in weight_dict:
                if weight_dict[k]==last_val_auc:
                    del weight_dict[k]
                    os.remove(k)
                    weight_dict[save_model_path]=val_auc
                    break
            torch.save({
                    "model":model.state_dict(),
                    "optimizer":model.optimizer.state_dict(),
                    "epoch":epoch
                },save_model_path)
            last_val_auc=min([weight_dict[k] for k in weight_dict])
        
        logger.info(log_text)
        
if __name__=='__main__':


    parser=argparse.ArgumentParser()
    parser.add_argument(dest='config')
    parser.add_argument('-n',dest='session_name')
    args=parser.parse_args()
    main(args)
        
