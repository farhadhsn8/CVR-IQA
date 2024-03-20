import torch
import os
from tqdm import tqdm

from dataloaders.dataloader_LQ_HQ_diff_content_HQ import DataLoader
from option_train_DistillationIQA import set_args, check_args
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
        'livec': list(range(0, 1162)),# no-ref image
        'koniq-10k': list(range(0, 10073)),# no-ref image
        'bid': list(range(0, 586)),# no-ref image
        'piq23': list(range(0, 5116)),# no-ref image  for all mode 
        'piq23_tr': list(range(0, 3630)),# no-ref image  for all mode 
        'piq23_ts': list(range(0, 1486)),# no-ref image  for all mode 

    }
folder_path = {
        'pipal':'./dataset/PIPAL',
        'live': './dataset/LIVE/',
        'csiq': './dataset/CSIQ/',
        'tid2013': './dataset/TID2013/',
        'livec': './dataset/LIVEC/',
        'koniq-10k': './dataset/koniq-10k/',
        'bid': './dataset/BID/',
        'kadid10k':'./dataset/kadid10k/',
        'piq23':'./dataset/PIQ23/'
    }



class DistillationIQASolver(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')
        self.txt_log_path = os.path.join(config.log_checkpoint_dir,'log.txt')
        self.config.studentNet_model_path = './model_zoo/stacking_ep98_saved_student_model.pth'
        with open(self.txt_log_path,"w+") as f:
            f.close()
        
        
        self.studentNet = DistillationIQANet(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer , stacking_mode=config.feature_stacking)
        if config.studentNet_model_path:
            self.studentNet._load_state_dict(torch.load(config.studentNet_model_path))
        print("studend model PATH:", config.studentNet_model_path)
        self.studentNet = self.studentNet.to(self.device)
        self.studentNet.train(True)
    
        #data
        test_loader_PIQ23_ts = DataLoader('piq23', folder_path['piq23'], config.ref_test_dataset_path, img_num['piq23_ts'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num ,  mode = "test20" , type="Overall")
        test_loader_LIVE = DataLoader('live', folder_path['live'], config.ref_test_dataset_path, img_num['live'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        test_loader_PIQ23 = DataLoader('piq23', folder_path['piq23'], config.ref_test_dataset_path, img_num['piq23'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num , mode = "all" , type="Overall")

        # test_loader_CLIVE = DataLoader('livec', folder_path['livec'], config.ref_test_dataset_path, img_num['livec'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        # test_loader_CSIQ = DataLoader('csiq', folder_path['csiq'], config.ref_test_dataset_path, img_num['csiq'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        # test_loader_TID = DataLoader('tid2013', folder_path['tid2013'], config.ref_test_dataset_path, img_num['tid2013'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        # test_loader_Koniq = DataLoader('koniq-10k', folder_path['koniq-10k'], config.ref_test_dataset_path, img_num['koniq-10k'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        
        self.test_data_LIVE = test_loader_LIVE.get_dataloader()
        self.test_data_PIQ23_ts = test_loader_PIQ23_ts.get_dataloader()
        self.test_data_PIQ23 = test_loader_PIQ23.get_dataloader()
        # self.test_data_CSIQ = test_loader_CSIQ.get_dataloader()
        # self.test_data_Koniq = test_loader_Koniq.get_dataloader()

    
    def write_list_to_file(self, list_data, file_path):
        with open(file_path, 'w') as filehandle:
            for listitem in list_data:
                filehandle.write('%s\n' % listitem)



    def test(self, test_data):
        self.studentNet.train(False)
        test_pred_scores, test_gt_scores = [], []
        for LQ_patches, _, ref_patches, label in tqdm(test_data):
            LQ_patches, ref_patches, label = LQ_patches.to(self.device), ref_patches.to(self.device), label.to(self.device)
            with torch.no_grad():
                _, _, pred = self.studentNet(LQ_patches, ref_patches)
                test_pred_scores.append(float(pred.item()))
                test_gt_scores = test_gt_scores + label.cpu().tolist()

        self.write_list_to_file(test_pred_scores , "pred_before.txt")
        self.write_list_to_file(test_gt_scores , "gt_before.txt")

        if self.config.use_fitting_prcc_srcc:
            fitting_pred_scores = convert_obj_score(test_pred_scores, test_gt_scores)
        test_pred_scores = np.mean(np.reshape(np.array(test_pred_scores), (-1, self.config.test_patch_num)), axis=1)
        test_gt_scores = np.mean(np.reshape(np.array(test_gt_scores), (-1, self.config.test_patch_num)), axis=1)
        self.write_list_to_file(test_pred_scores , "pred_after.txt")
        self.write_list_to_file(test_gt_scores , "gt_after.txt")
        test_srcc, _ = stats.spearmanr(test_pred_scores, test_gt_scores)
        if self.config.use_fitting_prcc_srcc:
            test_plcc, _ = stats.pearsonr(fitting_pred_scores, test_gt_scores)
        else:
            test_plcc, _ = stats.pearsonr(test_pred_scores, test_gt_scores)
        test_krcc, _ = stats.stats.kendalltau(test_pred_scores, test_gt_scores)
        test_srcc, test_plcc, test_krcc = abs(test_srcc), abs(test_plcc), abs(test_krcc)
        self.studentNet.train(True)
        return test_srcc, test_plcc, test_krcc

if __name__ == "__main__":
    config = set_args()
    config = check_args(config)
    solver = DistillationIQASolver(config=config)
    fold_10_test_LIVE_srcc, fold_10_test_LIVE_plcc, fold_10_test_LIVE_krcc = [], [], []
    fold_10_test_CLIVE_srcc, fold_10_test_CLIVE_plcc, fold_10_test_CLIVE_krcc = [], [], []
    fold_10_test_CSIQ_srcc, fold_10_test_CSIQ_plcc, fold_10_test_CSIQ_krcc = [], [], []
    fold_10_test_TID_srcc, fold_10_test_TID_plcc, fold_10_test_TID_krcc = [], [], []
    fold_10_test_Koniq_srcc, fold_10_test_Koniq_plcc, fold_10_test_Koniq_krcc = [], [], []
    fold_10_test_piq23ts_srcc, fold_10_test_piq23ts_plcc, fold_10_test_piq23ts_krcc = [], [], []
    fold_10_test_piq23_srcc, fold_10_test_piq23_plcc, fold_10_test_piq23_krcc = [], [], []

    for i in tqdm(range(1)):




        # test_CLIVE_srcc, test_CLIVE_plcc, test_CLIVE_krcc = solver.test(solver.test_data_CLIVE)
        # print('round{} Dataset:CLIVE Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(i, test_CLIVE_srcc, test_CLIVE_plcc, test_CLIVE_krcc))
        # fold_10_test_CLIVE_srcc.append(test_CLIVE_srcc)
        # fold_10_test_CLIVE_plcc.append(test_CLIVE_plcc)
        # fold_10_test_CLIVE_krcc.append(test_CLIVE_krcc)
        
        
        # test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc = solver.test(solver.test_data_CSIQ)
        # print('round{} Dataset:CSIQ Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(i, test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc))
        # fold_10_test_CSIQ_srcc.append(test_CSIQ_srcc)
        # fold_10_test_CSIQ_plcc.append(test_CSIQ_plcc)
        # fold_10_test_CSIQ_krcc.append(test_CSIQ_krcc)

        # test_TID_srcc, test_TID_plcc, test_TID_krcc = solver.test(solver.test_data_TID)
        # print('round{} Dataset:TID Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(i, test_TID_srcc, test_TID_plcc, test_TID_krcc))
        # fold_10_test_TID_srcc.append(test_TID_srcc)
        # fold_10_test_TID_plcc.append(test_TID_plcc)
        # fold_10_test_TID_krcc.append(test_TID_krcc)


        # test_Koniq_srcc, test_Koniq_plcc, test_Koniq_krcc = solver.test(solver.test_data_Koniq)
        # print('round{} Dataset:Koniq Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(i, test_Koniq_srcc, test_Koniq_plcc, test_Koniq_krcc))
        # fold_10_test_Koniq_srcc.append(test_Koniq_srcc)
        # fold_10_test_Koniq_plcc.append(test_Koniq_plcc)
        # fold_10_test_Koniq_krcc.append(test_Koniq_krcc)

        test_piq23_srcc, test_piq23_plcc, test_piq23_krcc = solver.test(solver.test_data_PIQ23)
        print('round{} Dataset:piq23 Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(i, test_piq23_srcc, test_piq23_plcc, test_piq23_krcc))
        fold_10_test_piq23_srcc.append(test_piq23_srcc)
        fold_10_test_piq23_plcc.append(test_piq23_plcc)
        fold_10_test_piq23_krcc.append(test_piq23_krcc)

        print("final!!")
        break

        # test_piq23ts_srcc, test_piq23ts_plcc, test_piq23ts_krcc = solver.test(solver.test_data_PIQ23_ts)
        # print('round{} Dataset:piq23-ts Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(i, test_piq23ts_srcc, test_piq23ts_plcc, test_piq23ts_krcc))
        # fold_10_test_piq23ts_srcc.append(test_piq23ts_srcc)
        # fold_10_test_piq23ts_plcc.append(test_piq23ts_plcc)
        # fold_10_test_piq23ts_krcc.append(test_piq23ts_krcc)

        # test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc = solver.test(solver.test_data_LIVE)
        # print('round{} Dataset:LIVE Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(i, test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc))
        # fold_10_test_LIVE_srcc.append(test_LIVE_srcc)
        # fold_10_test_LIVE_plcc.append(test_LIVE_plcc)
        # fold_10_test_LIVE_krcc.append(test_LIVE_krcc)




    
    print('Dataset:LIVE Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(np.mean(fold_10_test_LIVE_srcc), np.mean(fold_10_test_LIVE_plcc), np.mean(fold_10_test_LIVE_krcc)))
    print('Dataset:PIQ23-ts Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(np.mean(fold_10_test_piq23ts_srcc), np.mean(fold_10_test_piq23ts_plcc), np.mean(fold_10_test_piq23ts_krcc)))
    print('Dataset:PIQ23 Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(np.mean(fold_10_test_piq23_srcc), np.mean(fold_10_test_piq23_plcc), np.mean(fold_10_test_piq23_krcc)))

    # print('Dataset:CLIVE Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(np.mean(fold_10_test_CLIVE_srcc), np.mean(fold_10_test_CLIVE_plcc), np.mean(fold_10_test_CLIVE_krcc)))

    # print('Dataset:CSIQ Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(np.mean(fold_10_test_CSIQ_srcc), np.mean(fold_10_test_CSIQ_plcc), np.mean(fold_10_test_CSIQ_krcc)))
    # print('Dataset:TID Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(np.mean(fold_10_test_TID_srcc), np.mean(fold_10_test_TID_plcc), np.mean(fold_10_test_TID_krcc)))
    # print('Dataset:Koniq Test_SRCC:{} Test_PLCC:{} TEST_KRCC:{}\n'.format(np.mean(fold_10_test_Koniq_srcc), np.mean(fold_10_test_Koniq_plcc), np.mean(fold_10_test_Koniq_krcc)))






    
