import torch
import os
from tqdm import tqdm
import random
from dataloaders.dataloader_LQ_HQ_diff_content_HQ import DataLoader
from option_train_DistillationIQA import set_args, check_args
from scipy import stats
import numpy as np
from tools import convert_obj_score
from models.DistillationIQA import DistillationIQANet ,DistillationIQANet_makeRef


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





def write_to_file(list1 , file_name):
    with open(file_name, 'w') as filehandle:
        for listitem in list1:
            filehandle.write(str(listitem))
            filehandle.write(str(" , "))


class DistillationIQASolver(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if config.gpu_ids is not None else 'cpu')
        self.txt_log_path = os.path.join(config.log_checkpoint_dir,'log.txt')
        with open(self.txt_log_path,"w+") as f:
            f.close()
        

        self.studentNet = DistillationIQANet_makeRef(self_patch_num=config.self_patch_num, distillation_layer=config.distillation_layer , stacking_mode=config.feature_stacking)
        if config.studentNet_model_path:
            self.studentNet._load_state_dict(torch.load(config.studentNet_model_path))
        self.studentNet = self.studentNet.to(self.device)
        self.studentNet.train(True)

        #lr,opt,loss,epoch
        self.lr = config.lr
        self.lr_ratio = 1
        self.feature_loss_ratio = 1
        resnet_params = list(map(id, self.studentNet.feature_extractor.parameters()))
        res_params = filter(lambda p: id(p) not in resnet_params, self.studentNet.parameters())
        paras = [{'params': res_params, 'lr': self.lr * self.lr_ratio },
                {'params': self.studentNet.feature_extractor.parameters(), 'lr': self.lr}
                ]
        self.optimizer = torch.optim.Adam(paras, weight_decay=config.weight_decay)
        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.epochs = config.epochs

        #data
        config.train_index = img_num[config.train_dataset]
        random.shuffle(config.train_index)
        train_loader = DataLoader("piq23",  folder_path['piq23'], config.ref_train_dataset_path, img_num['piq23_tr'], config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True, self_patch_num=config.self_patch_num ,  mode = "train80" , type="Overall")
        test_loader_LIVE = DataLoader('live', folder_path['live'], config.ref_test_dataset_path, img_num['live'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        # test_loader_CSIQ = DataLoader('csiq', folder_path['csiq'], config.ref_test_dataset_path, img_num['csiq'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        test_loader_PIQ23_ts = DataLoader('piq23', folder_path['piq23'], config.ref_test_dataset_path, img_num['piq23_ts'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num ,  mode = "test20" , type="Overall")
        # test_loader_TID = DataLoader('tid2013', folder_path['tid2013'], config.ref_test_dataset_path, img_num['tid2013'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        # test_loader_Koniq = DataLoader('koniq-10k', folder_path['koniq-10k'], config.ref_test_dataset_path, img_num['koniq-10k'], config.patch_size, config.test_patch_num, istrain=False, self_patch_num=config.self_patch_num)
        
        self.train_data = train_loader.get_dataloader()
        self.test_data_LIVE = test_loader_LIVE.get_dataloader()
        # self.test_data_CSIQ = test_loader_CSIQ.get_dataloader()
        self.test_data_PIQ23_ts = test_loader_PIQ23_ts.get_dataloader()
        # self.test_data_TID = test_loader_TID.get_dataloader()
        # self.test_data_Koniq = test_loader_Koniq.get_dataloader()


    def train(self):
        best_srcc_LIVE, best_srcc_CSIQ, best_srcc_TID, best_srcc_Koniq = 0.0, 0.0, 0.0, 0.0
        best_plcc_LIVE, best_plcc_CSIQ, best_plcc_TID, best_plcc_Koniq = 0.0, 0.0, 0.0, 0.0
        best_krcc_LIVE, best_krcc_CSIQ, best_krcc_TID, best_krcc_Koniq = 0.0, 0.0, 0.0, 0.0


        
        total_params = sum(p.numel() for p in self.studentNet.parameters())
        print(f"\n\nNumber of total parameters in student net: {total_params}")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.studentNet.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in student net: {trainable_params}\n\n")


        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_KRCC')

        # NEW
        scaler = torch.cuda.amp.GradScaler()
        test_TID_srcc, test_TID_plcc, test_TID_krcc = 0 , 0, 0
        featureLosses = []
        predLosses = []
        sumLosses = []
        train_acc = []
        test_acc = []
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for LQ_patches, _ , ref_patches, label in tqdm(self.train_data):
                LQ_patches,  ref_patches, label = LQ_patches.to(self.device),  ref_patches.to(self.device), label.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    _, _, pred = self.studentNet(LQ_patches, ref_patches)
                
                    pred_scores = pred_scores + pred.cpu().tolist()
                    gt_scores = gt_scores + label.cpu().tolist()
                    loss = self.l1_loss(pred.squeeze(), label.float().detach())

                  

                    predLosses.append(loss)


                epoch_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            
            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            train_acc.append(train_srcc)
            if t % 5 ==0:
                test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc = self.test(self.test_data_LIVE)
                # test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc = self.test(self.test_data_CSIQ)
                test_piq_srcc, test_piq_plcc, test_piq_krcc = self.test(self.test_data_PIQ23_ts)
                # -----------> test piq23 validation <-----------
            # test_acc.append(test_TID_srcc)
            # test_Koniq_srcc, test_Koniq_plcc, test_Koniq_krcc = solver.test(solver.test_data_Koniq)

            # if test_LIVE_srcc + test_LIVE_plcc + test_LIVE_krcc > best_srcc_LIVE + best_plcc_LIVE + best_krcc_LIVE:
            #     best_srcc_LIVE, best_srcc_CSIQ, best_srcc_TID = test_LIVE_srcc, test_CSIQ_srcc, test_TID_srcc
            #     print('%d:live\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            #     (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc))

            # if test_CSIQ_srcc + test_CSIQ_plcc + test_CSIQ_krcc > best_srcc_CSIQ + best_plcc_CSIQ + best_krcc_CSIQ:
            #     best_plcc_LIVE, best_plcc_CSIQ, best_plcc_TID = test_LIVE_plcc, test_CSIQ_plcc, test_TID_plcc
            #     print('%d:csiq\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            #     (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc))

            # if test_TID_srcc + test_TID_plcc + test_TID_krcc > best_srcc_TID + best_plcc_TID + best_krcc_TID:
            #     best_srcc_TID, best_plcc_TID, best_krcc_TID = test_TID_srcc, test_TID_plcc, test_TID_krcc

            print('%d:live\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_LIVE_srcc, test_LIVE_plcc, test_LIVE_krcc))

            # print('%d:csiq\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            # (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_CSIQ_srcc, test_CSIQ_plcc, test_CSIQ_krcc))

            print('%d:piq\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_piq_srcc, test_piq_plcc, test_piq_krcc))

            # if test_Koniq_srcc + test_Koniq_plcc + test_Koniq_krcc > best_srcc_Koniq + best_plcc_Koniq + best_krcc_Koniq:
            #     print('%d:koniq-10k\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f \n' %
            #     (t, sum(epoch_loss) / len(epoch_loss), train_srcc, test_Koniq_srcc, test_Koniq_plcc, test_Koniq_krcc))
            #     best_srcc_Koniq, best_plcc_Koniq, best_krcc_Koniq = test_Koniq_srcc, test_Koniq_plcc, test_Koniq_krcc


            torch.save(self.studentNet.state_dict(), os.path.join(self.config.model_checkpoint_dir, 'NR(make-ref)_{}_saved_model.pth'.format(t)))
            
            self.lr = self.lr / pow(10, (t // self.config.update_opt_epoch))
            if t > 20:
                self.lr_ratio = 1
            resnet_params = list(map(id, self.studentNet.feature_extractor.parameters()))
            rest_params = filter(lambda p: id(p) not in resnet_params, self.studentNet.parameters())
            paras = [{'params': rest_params, 'lr': self.lr * self.lr_ratio },
                    {'params': self.studentNet.feature_extractor.parameters(), 'lr': self.lr}
                    ]
            self.optimizer = torch.optim.Adam(paras, weight_decay=self.config.weight_decay)
        # print('Best live test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_LIVE, best_plcc_LIVE, best_krcc_LIVE))
        # print('Best csiq test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_CSIQ, best_plcc_CSIQ, best_krcc_CSIQ))
        # print('Best tid2013 test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_TID, best_plcc_TID, best_krcc_TID))
        # print('Best koniq-10k test SRCC %f, PLCC %f, KRCC %f\n' % (best_srcc_Koniq, best_plcc_Koniq, best_krcc_Koniq))


    def test(self, test_data):
        self.studentNet.train(False)
        test_pred_scores, test_gt_scores = [], []
        for LQ_patches, _, ref_patches, label in tqdm(test_data):
            LQ_patches, ref_patches, label = LQ_patches.to(self.device), ref_patches.to(self.device), label.to(self.device)
            with torch.no_grad():
                _, _, pred = self.studentNet(LQ_patches, ref_patches)
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
        self.studentNet.train(True)
        return test_srcc, test_plcc, test_krcc

if __name__ == "__main__":
    config = set_args()
    config = check_args(config)
    solver = DistillationIQASolver(config=config)
    solver.train()




    
