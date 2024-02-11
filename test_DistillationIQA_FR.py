import torch
import os
from tqdm import tqdm

from dataloaders.dataloader_LQ_HQ_diff_content_HQ import DataLoader
from option_train_DistillationIQA_FR import set_args, check_args
from scipy import stats
import numpy as np
from tools import convert_obj_score
from models.DistillationIQA import DistillationIQANet 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


img_num = {
        'kadid10k': list(range(0,10125)),
        'live': list(range(0, 29)),#ref HR image
        'csiq': list(range(0, 30)),#ref HR image
        'tid2013': list(range(0, 25)),
    }
folder_path = {
        'pipal':'./dataset/PIPAL',
        'live': './dataset/live/',
        'csiq': './dataset/CSIQ/',
        'tid2013': './dataset/TID2013/',
        'livec': './dataset/LIVEC/',
        'koniq-10k': './dataset/koniq-10k/',
        'bid': './dataset/BID/',
        'kadid10k':'./dataset/kadid10k/'
    }


class DistillationIQASolver(object):
    def __init__(self, config):
        config.batch_size = 1
        self.config = config

        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')
        self.txt_log_path = os.path.join(config.log_checkpoint_dir,'log.txt')
        self.config.teacherNet_model_path = './model_zoo/BEST_FRIQA_54_saved_model.pth'
        with open(self.txt_log_path,"w+") as f:
            f.close()
        
        #model
        self.teacherNet = DistillationIQANet(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer ,  stacking_mode=config.feature_stacking)
        if config.teacherNet_model_path:
            print(config.teacherNet_model_path)
            self.teacherNet._load_state_dict(torch.load(config.teacherNet_model_path))
        self.teacherNet = self.teacherNet.to(self.device)
        self.teacherNet.train(False)
       
    
        #data
        test_loader_CSIQ = DataLoader('csiq', folder_path['csiq'], config.ref_test_dataset_path, img_num['csiq'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num, batch_size=self.config.batch_size )
        test_loader_LIVE = DataLoader('live', folder_path['live'], config.ref_test_dataset_path, img_num['live'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num, batch_size=self.config.batch_size)
        test_loader_TID = DataLoader('tid2013', folder_path['tid2013'], config.ref_test_dataset_path, img_num['tid2013'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num, batch_size=self.config.batch_size)
        
        self.test_data_CSIQ = test_loader_CSIQ.get_dataloader()
        self.test_data_LIVE = test_loader_LIVE.get_dataloader()
        self.test_data_TID = test_loader_TID.get_dataloader()

    
    def test(self, test_data):
        self.teacherNet.train(False)
        test_pred_scores, test_gt_scores = [], []
        for LQ_patches, HQ_patches, diff_patches, label in tqdm(test_data):
            LQ_patches, HQ_patches, label = LQ_patches.to(self.device), HQ_patches.to(self.device), label.to(self.device)
            with torch.no_grad():
                _, _, pred = self.teacherNet(LQ_patches, HQ_patches)
                test_pred_scores.append(float(pred.item()))
                test_gt_scores = test_gt_scores + label.cpu().tolist()
        if self.config.use_fitting_prcc_srcc:
            fitting_pred_scores = convert_obj_score(test_pred_scores, test_gt_scores)
        test_pred_scores = np.mean(np.reshape(np.array(test_pred_scores), (-1, self.config.test_patch_num)), axis=1)
        test_gt_scores = np.mean(np.reshape(np.array(test_gt_scores), (-1, self.config.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(test_pred_scores, test_gt_scores)
        if self.config.use_fitting_prcc_srcc:
            test_plcc, _ = stats.pearsonr(fitting_pred_scores, test_gt_scores)
        else:
            test_plcc, _ = stats.pearsonr(test_pred_scores, test_gt_scores)
        test_krcc, _ = stats.stats.kendalltau(test_pred_scores, test_gt_scores)
        test_srcc, test_plcc, test_krcc = abs(test_srcc), abs(test_plcc), abs(test_krcc)
        self.teacherNet.train(True)
        return test_srcc, test_plcc, test_krcc

if __name__ == "__main__":
    config = set_args()
    config = check_args(config)
    solver = DistillationIQASolver(config=config)
    fold_10_test_LIVE_srcc, fold_10_test_LIVE_plcc, fold_10_test_LIVE_krcc = [], [], []
    fold_10_test_CSIQ_srcc, fold_10_test_CSIQ_plcc, fold_10_test_CSIQ_krcc = [], [], []
    fold_10_test_TID_srcc, fold_10_test_TID_plcc, fold_10_test_TID_krcc = [], [], []

    for i in tqdm(range(3)):

        
        test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc = solver.test(solver.test_data_CSIQ)
        print('round{} Dataset:CSIQ Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(i, test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc))
        fold_10_test_CSIQ_srcc.append(test_CSIQ_srcc)
        fold_10_test_CSIQ_plcc.append(test_CSIQ_plcc)
        fold_10_test_CSIQ_krcc.append(test_CSIQ_krcc)



        test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc = solver.test(solver.test_data_LIVE)
        print('round{} Dataset:LIVE Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(i, test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc))
        fold_10_test_LIVE_srcc.append(test_LIVE_srcc)
        fold_10_test_LIVE_plcc.append(test_LIVE_plcc)
        fold_10_test_LIVE_krcc.append(test_LIVE_krcc)


        
        test_TID_srcc, test_TID_plcc, test_TID_krcc = solver.test(solver.test_data_TID)
        print('round{} Dataset:TID Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(i, test_TID_srcc, test_TID_plcc, test_TID_krcc))
        fold_10_test_TID_srcc.append(test_TID_srcc)
        fold_10_test_TID_plcc.append(test_TID_plcc)
        fold_10_test_TID_krcc.append(test_TID_krcc)
        


    
    print('Dataset:LIVE Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(np.mean(fold_10_test_LIVE_srcc), np.mean(fold_10_test_LIVE_plcc), np.mean(fold_10_test_LIVE_krcc)))
    print('Dataset:CSIQ Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(np.mean(fold_10_test_CSIQ_srcc), np.mean(fold_10_test_CSIQ_plcc), np.mean(fold_10_test_CSIQ_krcc)))
    print('Dataset:TID Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(np.mean(fold_10_test_TID_srcc), np.mean(fold_10_test_TID_plcc), np.mean(fold_10_test_TID_krcc)))






    
