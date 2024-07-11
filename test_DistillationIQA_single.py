import torch
import os
from option_train_DistillationIQA import set_args, check_args
import numpy as np
from models.DistillationIQA import DistillationIQANet_org_or_stackingV1 , DistillationIQANet_org_or_stackingV2
from PIL import Image
import torchvision


img_num = {
        'kadid10k': list(range(0,10125)),
        'live': list(range(0, 29)),#ref HR image
        'csiq': list(range(0, 30)),#ref HR image
        'tid2013': list(range(0, 25)),
        'livec': list(range(0, 1162)),# no-ref image
        'koniq-10k': list(range(0, 10073)),# no-ref image
        'bid': list(range(0, 586)),# no-ref image
    }
folder_path = {
        'pipal':'./dataset/PIPAL',
        'live': './dataset/LIVE/',
        'csiq': './dataset/CSIQ/',
        'tid2013': './dataset/TID2013/',
        'livec': './dataset/LIVEC/',
        'koniq-10k': './dataset/koniq-10k/',
        'bid': './dataset/BID/',
        'kadid10k':'./dataset/kadid10k/'
    }


class DistillationIQASolver(object):
    def __init__(self, student_address , net_mode):
        config = set_args()
        self.config = config
        # self.config.teacherNet_model_path = './model_zoo/FR_teacher_cross_dataset.pth'
        self.config.studentNet_model_path = student_address



        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')
        self.txt_log_path = os.path.join(config.log_checkpoint_dir,'log.txt')
        with open(self.txt_log_path,"w+") as f:
            f.close()
        
        #model
        # self.teacherNet = DistillationIQANet(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer)
        # if config.teacherNet_model_path:
        #     self.teacherNet._load_state_dict(torch.load(config.teacherNet_model_path))
        # self.teacherNet = self.teacherNet.to(self.device)
        # self.teacherNet.train(False)
        if net_mode == "org":
            self.studentNet = DistillationIQANet_org_or_stackingV1(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer, stacking_mode=False)
        if net_mode == "stackingV1":
            self.studentNet = DistillationIQANet_org_or_stackingV1(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer, stacking_mode=True)
        if net_mode == "stackingV2":
            self.studentNet = DistillationIQANet_org_or_stackingV2(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer, stacking_mode=True)
        
        if config.studentNet_model_path:
            self.studentNet._load_state_dict(torch.load(config.studentNet_model_path))
            print(">>>>> ",config.studentNet_model_path)
        self.studentNet = self.studentNet.to(self.device)
        self.studentNet.train(True)

        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=self.config.patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
        #data
      
    
    def preprocess(self, path , resize=False):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if True == resize:
                img = img.resize((224, 224))
            img= img.convert('RGB')
        patches = []
        for _ in range(self.config.self_patch_num):
            patch = self.transform(img)
            patches.append(patch.unsqueeze(0))
        patches = torch.cat(patches, 0)
        return patches.unsqueeze(0)

    def test(self, lq_path, ref_path):
        self.LQ_patches = self.preprocess(lq_path , resize=False)
        self.ref_patches = self.preprocess(ref_path , resize=False)
        self.studentNet.train(False)
        LQ_patches, ref_patches = self.LQ_patches.to(self.device), self.ref_patches.to(self.device)
        with torch.no_grad():
            _, _, pred = self.studentNet(LQ_patches, ref_patches)
        return float(pred.item())
    

    def ref_features_before_minus(self, ref_path):
        self.ref_patches = self.preprocess(ref_path)
        self.studentNet.train(False)
        ref_patches =  self.ref_patches.to(self.device)
        with torch.no_grad():
           f = np.array(self.studentNet.ref_features_before_minus(ref_patches, ref_patches)[0].cpu())
        #    f[0],f[1],f[2],f[3] = f[0].cpu() , f[1].cpu(),f[2].cpu(),f[3].cpu()
        return f
    


    def cvr_on_single_image(self , lq_path , ref_path  ):
        scores = []
        for _ in range(1):
            scores.append(self.test(lq_path=lq_path, ref_path=ref_path))
        return (np.mean(scores))
    






    


    
