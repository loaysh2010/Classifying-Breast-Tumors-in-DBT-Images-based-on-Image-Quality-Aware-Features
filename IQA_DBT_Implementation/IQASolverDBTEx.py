import torch
from scipy import stats
from Myfolders import DBTexFolder
import torchvision
from network import *
import models_H


class IQASolver(object):
    """Solver for training and testing hyperIQA"""
    def __init__(self, config, path, train_idx,test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num

        # self.model_hyper = models_H.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
        # self.model_hyper.train(True)

        self.model = SimpleNet().cuda()
        # self.model = load_ResNet50().cuda()
        # self.model = load_convnext().cuda()

        self.model.train(True)

        self.l1_loss = torch.nn.L1Loss().cuda()

        self.solver = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # self.solver = torch.optim.AdamW(self.model.parameters(),lr=4e-3)


        # backbone_params = list(map(id, self.model_hyper.res.parameters()))
        # self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model_hyper.parameters())
        # self.lr = config.lr
        # self.lrratio = config.lr_ratio
        # self.weight_decay = config.weight_decay
        # paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
        #          {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
        #          ]
        # self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)



        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(size=config.patch_size),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
            #                                  std=(0.229, 0.224, 0.225))
        ])

        Train_data = DBTexFolder(root=path+'Train', index=train_idx, transform=transforms, patch_num=config.train_patch_num)
        Test_data = DBTexFolder(root=path + 'Test', index=test_idx, transform=transforms, patch_num=config.train_patch_num)

        self.train_data = torch.utils.data.DataLoader(Train_data, batch_size=config.batch_size, shuffle=True)
        self.test_data = torch.utils.data.DataLoader(Test_data, batch_size=1, shuffle=True)
        # train_loader = my_data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size, config.train_patch_num, batch_size=config.batch_size, istrain=True)
        # test_loader = my_data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size, config.test_patch_num, istrain=False)
        # val_loader = data_loader.DataLoader(config.dataset, path, val_idx, config.patch_size, config.test_patch_num,  istrain=False)

        # self.train_data = train_loader.get_data()
        # self.val_data = val_loader.get_data()
        # self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        best_srcc = 0.0
        best_plcc = 0.0
        print('Epoch\tTrain_Loss\tTrain_SRCC\t|| \tTest_SRCC\tTest_PLCC')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []

            for img, label in self.train_data:

                img = img.cuda()
                label = (label).cuda()
                # label = (label/100).cuda() # LIVE
                # label = (label / 10).cuda() #TID2013

                self.solver.zero_grad()

                pred = self.model(img)

                # # Generate weights for target network
                # paras = self.model_hyper(img)  # 'paras' contains the network weights conveyed to target network
                #
                # # Building target network
                # model_target = models_H.TargetNet(paras).cuda()
                # for param in model_target.parameters():
                #     param.requires_grad = False
                #
                # # Quality prediction
                # pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net

                if type(pred.cpu().tolist()) == type(list()):
                    pred_scores = pred_scores + (pred).cpu().tolist()
                    gt_scores = gt_scores + (label).cpu().tolist()

                    # # LIVE
                    # pred_scores = pred_scores + (pred*100).cpu().tolist()
                    # gt_scores = gt_scores + (label*100).cpu().tolist()

                    # TID2013
                    # pred_scores = pred_scores + (pred * 10).cpu().tolist()
                    # gt_scores = gt_scores + (label * 10).cpu().tolist()
                else:
                    continue

                loss = self.l1_loss(pred.squeeze(), label.float().detach())
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            # train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

            test_srcc, test_plcc = self.test(self.test_data)
            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                torch.save(self.model.state_dict(), 'Checkpoints/SimpleNetlarsh20_DBTex.pt')
                # torch.save(self.model_hyper.state_dict(), 'Checkpoints/ConvNext_DBTex.pt')
            print('\t%d\t\t%4.3f\t\t%4.4f\t\t || %4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc))

            # # Update optimizer
            # lr = self.lr / pow(10, (t // 6))
            # if t > 8:
            #     self.lrratio = 1
            # self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
            #               {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
            #               ]
            # self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        print('Best test SRCC %f, PLCC %f' % (best_srcc, best_plcc))

        return best_srcc, best_plcc

    def test(self, data):
        """Testing"""
        # self.model_hyper.train(False)
        self.model.train(False)
        pred_scores = []
        gt_scores = []
        with torch.no_grad():
            for img, label in data:
                # Data.
                # img = torch.tensor(img.cuda())
                # label = torch.tensor(label.cuda())
                img = img.cuda()
                label = (label).cuda()
                # label = (label/100).cuda() #LIVE
                # label = (label / 10).cuda() #TID2013

                # paras = self.model_hyper(img)
                # model_target = models_H.TargetNet(paras).cuda()
                # model_target.train(False)
                # pred = model_target(paras['target_in_vec'])

                pred = self.model(img)

                pred_scores.append(float(pred.item()))
                gt_scores = gt_scores + label.cpu().tolist()

                # # LIVE
                # pred_scores.append(float(pred.item()*100))
                # gt_scores = gt_scores + (label*100).cpu().tolist()

                # TID2013
                # pred_scores.append(float(pred.item() * 10))
                # gt_scores = gt_scores + (label * 10).cpu().tolist()

        # pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        # gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        # self.model_hyper.train(True)
        self.model.train(True)
        return test_srcc, test_plcc