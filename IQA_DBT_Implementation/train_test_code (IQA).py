import random
import numpy as np

from IQASolverDBTEx import IQASolver

class Configration():
    def __init__(self):
        self.dataset =  'DBTex'  # 'tid2013'
        self.patch_size = 224
        self.patchs_per_image = 25
        self.train_test_num = 10
        self.epochs = 30
        self.train_patch_num = 25
        self.test_patch_num = 25
        self.batch_size = 32
        self.lr = 2e-5
        self.lr_ratio = 10
        self.weight_decay = 5e-4

def main(config):

    folder_path = {
        'live': '../Data/LIVE/',
        'tid2013': '../Data/tid2013/',
        'livec': '/home/ssl/Database/ChallengeDB_release/ChallengeDB_release/',
        'DBTex': 'DATA/DBTex_Data_3_crop_KeySlice/'
    }
    # img_num = {
    #     'live': list(range(0, 29)),
    #     'tid2013': list(range(0, 25)),
    #     'livec': list(range(0, 1162)),
    # }


    path = folder_path[config.dataset]

    # sel_num = img_num[config.dataset]

    # srcc_all = np.zeros(config.train_test_num, dtype=float)
    # plcc_all = np.zeros(config.train_test_num, dtype=float)
    # print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    # for i in range(config.train_test_num):
    #     print('Round %d' % (i + 1))
        # Randomly select 80% images for training and the rest for testing
    print('Begin Training')
    train_index = list(range(0,200))
    test_index = list(range(0,50))
    random.shuffle(train_index)
    random.shuffle(test_index)


    solver = IQASolver(config, path, train_index, test_index)
    srcc_med, plcc_med  = solver.train()

        # print(srcc_all)
        # print(plcc_all)
    # srcc_med = np.median(srcc_all)
    # plcc_med = np.median(plcc_all)

    print('Testing median SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))

if __name__ == '__main__':
    config = Configration()
    main(config)

