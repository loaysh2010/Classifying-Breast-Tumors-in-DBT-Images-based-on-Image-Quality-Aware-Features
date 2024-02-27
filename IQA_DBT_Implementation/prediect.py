import os
import cv2
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from network import *
from torchvision import transforms
from cv2 import matchTemplate as cv2m
from matplotlib import  pyplot as plt
from sklearn.metrics import mean_squared_error
import math
#=================================#
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img
def getDBTexName(path, suffix):
    filename = []
    f_list = sorted(os.listdir(path))
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i.split(suffix)[0])
    return filename
def search_sequence_cv2(arr,seq):
    """ Find sequence in an array using cv2.
    """

    # Run a template match with input sequence as the template across
    # the entire length of the input array and get scores.
    S = cv2m(arr.astype('uint8'),seq.astype('uint8'),cv2.TM_SQDIFF)

    # Now, with floating point array cases, the matching scores might not be
    # exactly zeros, but would be very small numbers as compared to others.
    # So, for that use a very small to be used to threshold the scorees
    # against and decide for matches.
    thresh = 1e-5 # Would depend on elements in seq. So, be careful setting this.

    # Find the matching indices
    idx = np.where(S.ravel() < thresh)[0]

    # Get the range of those indices as final output
    if len(idx)>0:
        return np.unique((idx[:,None] + np.arange(seq.size)).ravel())
    else:
        return []
def crop_Breast(source_img,view):
    views = ['lcc', 'lmlo', 'rcc', 'rmlo']
    rand_view = views.index(view)
    h, w = source_img[:,:,0].shape
    h_c, w_c = int(h / 2), int(w / 2)
    if rand_view == 0 or rand_view == 1:
        tw = source_img[h_c, :].reshape(1, -1)
        idx_w = search_sequence_cv2(tw, np.zeros((1, 100)))

        th = source_img[:, 0].reshape(1, -1)
        th_1 = th[:, 0:h_c]
        th_2 = th[:, h_c:h]
        idx_h1 = search_sequence_cv2(th_1, np.zeros((1, 100)))
        idx_h2 = search_sequence_cv2(th_2, np.zeros((1, 100)))

        if len(idx_h1) == 0:
            idx_h1 = [0]
        if len(idx_h2) == 0:
            idx_h2 = [h_c]

        if rand_view == 0:
            source_img = source_img[idx_h1[-1]:h_c + idx_h2[0], 0:idx_w[20]]
        else:
            source_img = source_img[idx_h1[-1]:h_c + idx_h2[0], 0:idx_w[100]]
        # plt.imshow(source_img,cmap='gray')
        # plt.show()
    elif rand_view == 2 or rand_view == 3:
        tw = source_img[h_c, :].reshape(1, -1)
        idx_w = search_sequence_cv2(tw, np.zeros((1, 100)))

        th = source_img[:, w - 1].reshape(1, -1)
        th_1 = th[:, 0:h_c]
        th_2 = th[:, h_c:h]
        idx_h1 = search_sequence_cv2(th_1, np.zeros((1, 100)))
        idx_h2 = search_sequence_cv2(th_2, np.zeros((1, 100)))

        if len(idx_h1) == 0:
            idx_h1 = [0]
        if len(idx_h2) == 0:
            idx_h2 = [h_c]

        if rand_view == 0:
            source_img = source_img[idx_h1[-1]:h_c + idx_h2[0], idx_w[-1] + 80:w]
        else:
            source_img = source_img[idx_h1[-1]:h_c + idx_h2[0], idx_w[-1]:w]
    return source_img
#====================================#
def get_data(root):
    imgpath = os.path.join(root, '03_Distorted_Images')
    imgnames = getDBTexName(imgpath, '.png')


    target = []
    with open(root.split('Test')[0] + 'labels.pkl', 'rb') as handle:
        labels = pickle.load(handle)
    for name in imgnames:
        target.append(labels[name].pop())
    labels = np.array(target).astype(np.float32)

    transform = transforms.ToTensor()
    samples=torch.ones(50,25,3,224,224)
    for s,img_name in enumerate(imgnames):
        img = pil_loader(os.path.join(root,'03_Distorted_Images', img_name+'.png'))
        img = np.asarray(img)
        img_list=torch.ones(25,3,224,224)
        for i in range(25):
            xidx = np.random.randint(0,img.shape[1]-224)
            yidx = np.random.randint(0, img.shape[0] - 224)
            im = img[yidx:yidx+224,xidx:xidx+224,:]
            img_list[i]=transform(im)
        img_list = torch.Tensor(img_list)
        samples[s]=img_list
    # samples = torch.Tensor(samples)
    # samples = transform(samples)

    return imgnames,samples,labels
def get_data_test(root):

    # imgpath = os.path.join(root, 'images')
    imgpath = os.path.join(root, '03_Distorted_Images')
    imgnames = getDBTexName(imgpath, '.png')

    # target = []
    # with open(root.split('Test')[0] + 'labels.pkl', 'rb') as handle:
    #     labels = pickle.load(handle)
    # for name in imgnames:
    #     target.append(labels[name].pop())
    # labels = np.array(target).astype(np.float32)

    transform = transforms.ToTensor()
    samples = torch.ones(len(imgnames), 25, 3, 224, 224)
    for s, img_name in enumerate(imgnames):
        # img = pil_loader(os.path.join(imgpath, img_name + '.png'))
        # img = np.asarray(img)
        img = cv2.imread(os.path.join(imgpath, img_name + '.png'))
        img_list = torch.ones(25, 3, 224, 224)
        for i in range(25):
            xidx = np.random.randint(0, img.shape[1] - 224)
            yidx = np.random.randint(0, img.shape[0] - 224)
            im = img[yidx:yidx + 224, xidx:xidx + 224, :]
            img_list[i] = transform(im)
        img_list = torch.Tensor(img_list)
        samples[s] = img_list

    return imgnames, samples
def get_data_keys(root):
    imgnames = sorted(os.listdir(root))
    transform = transforms.ToTensor()
    samples = torch.ones(len(imgnames), 25, 3, 224, 224)
    for s, img_name in enumerate(imgnames):
        # img = pil_loader(os.path.join(imgpath, img_name + '.png'))
        # img = np.asarray(img)
        views = ['lcc', 'lmlo', 'rcc', 'rmlo']
        rand_view = views.index(img_name.split('_')[1])
        img = cv2.imread(root+img_name)
        source_img = img[:, :, 0]  # /255.0
        # h, w = source_img.shape
        # h_c, w_c = int(h / 2), int(w / 2)
        #
        # if rand_view == 0 or rand_view == 1:
        #     tw = source_img[h_c, :].reshape(1, -1)
        #     idx_w = search_sequence_cv2(tw, np.zeros((1, 100)))
        #
        #     th = source_img[:, 0].reshape(1, -1)
        #     th_1 = th[:, 0:h_c]
        #     th_2 = th[:, h_c:h]
        #     idx_h1 = search_sequence_cv2(th_1, np.zeros((1, 100)))
        #     idx_h2 = search_sequence_cv2(th_2, np.zeros((1, 100)))
        #
        #     if len(idx_h1) == 0:
        #         idx_h1 = [0]
        #     if len(idx_h2) == 0:
        #         idx_h2 = [h_c]
        #
        #     if rand_view == 0:
        #         source_img = source_img[idx_h1[-1]:h_c + idx_h2[0], 0:idx_w[0] + 20]
        #     else:
        #         source_img = source_img[idx_h1[-1]:h_c + idx_h2[0], 0:idx_w[0] + 100]
        #     # plt.imshow(source_img,cmap='gray')
        #     # plt.show()
        # elif rand_view == 2 or rand_view == 3:
        #     tw = source_img[h_c, :].reshape(1, -1)
        #     idx_w = search_sequence_cv2(tw, np.zeros((1, 100)))
        #
        #     th = source_img[:, w - 1].reshape(1, -1)
        #     th_1 = th[:, 0:h_c]
        #     th_2 = th[:, h_c:h]
        #     idx_h1 = search_sequence_cv2(th_1, np.zeros((1, 100)))
        #     idx_h2 = search_sequence_cv2(th_2, np.zeros((1, 100)))
        #
        #     if len(idx_h1) == 0:
        #         idx_h1 = [0]
        #     if len(idx_h2) == 0:
        #         idx_h2 = [h_c]
        #
        #     if rand_view == 0:
        #         source_img = source_img[idx_h1[-1]:h_c + idx_h2[0], idx_w[-1] + 80:w]
        #     else:
        #         source_img = source_img[idx_h1[-1]:h_c + idx_h2[0], idx_w[-1]:w]
        source_img = source_img/255.0
        # source_img = img/255.0
        img_list = torch.ones(25, 3, 224, 224)
        for i in range(25):
            # xidx = np.random.randint(0, img.shape[1] - 224)
            # yidx = np.random.randint(0, img.shape[0] - 224)
            # im = img[yidx:yidx + 224, xidx:xidx + 224, :]
            xidx = np.random.randint(0, source_img.shape[1] - 224)
            yidx = np.random.randint(0, source_img.shape[0] - 224)
            im = source_img[yidx:yidx + 224, xidx:xidx + 224]
            im = np.concatenate((np.expand_dims(im, 2), np.expand_dims(im, 2), np.expand_dims(im, 2)), 2)
            img_list[i] = transform(im)
        img_list = torch.Tensor(img_list)
        samples[s] = img_list
    return imgnames, samples
def get_data_general(root):
    imgnames = sorted(os.listdir(root))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                          ])
    # samples = torch.ones(len(imgnames), 25, 3, 224, 224)
    img_list = torch.ones((len(imgnames)), 3, 224, 224)
    for s, img_name in enumerate(imgnames):

        img = cv2.imread(root+img_name)
        img = img/255.0
        # im = np.concatenate((source_img, source_img, source_img), 2)
        img_list[s] = transform(img)
        img_list = torch.Tensor(img_list)
        # samples[s] = img_list
    return imgnames, img_list
#==========================================#
def plot_scatter(actual_data,predict_data):
    actual_data = np.asarray(actual_data)
    predict_data = np.asarray(predict_data)

    plt.scatter(actual_data,predict_data,label='ResNet')
    plt.ylabel('Objective quality score')
    plt.xlabel('predicted score')
    plt.legend()
    plt.show()
def plot_scatter_all ():
    with open('ResNet50.pkl', 'rb') as handle:
        ResNet50 = pickle.load(handle)
    with open('SimpleNet.pkl', 'rb') as handle:
        SimpleNet = pickle.load(handle)
    with open('HyperNet.pkl', 'rb') as handle:
        Hyper = pickle.load(handle)
    with open('ConvNext.pkl', 'rb') as handle:
        ConvNext = pickle.load(handle)

    lineStart = min(ResNet50['Actual_scores'])
    lineEnd   = max(ResNet50['Actual_scores'])
    plt.scatter(ResNet50['Actual_scores'], ResNet50['Predicted'], label='ResNet50')
    plt.scatter(SimpleNet['Actual_scores'], SimpleNet['Predicted'], label='SimpleNet')
    plt.scatter(Hyper['Actual_scores'], Hyper['Predicted'], label='Hyper Network')
    plt.scatter(ConvNext['Actual_scores'], ConvNext['Predicted'], label='ConvNext')
    plt.ylabel('Objective quality score')
    plt.xlabel('predicted score')
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color='k')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.legend()
    plt.show()
def calculate_MSE():
    with open('ResNet50.pkl', 'rb') as handle:
        ResNet50 = pickle.load(handle)
    with open('SimpleNet.pkl', 'rb') as handle:
        SimpleNet = pickle.load(handle)
    with open('HyperNet.pkl', 'rb') as handle:
        Hyper = pickle.load(handle)
    with open('ConvNext.pkl', 'rb') as handle:
        ConvNext = pickle.load(handle)

    MSE_ResNet = mean_squared_error(ResNet50['Actual_scores'], ResNet50['Predicted'])
    MSE_SimpleNet = mean_squared_error(SimpleNet['Actual_scores'], SimpleNet['Predicted'])
    MSE_Hyper = mean_squared_error(Hyper['Actual_scores'], Hyper['Predicted'])
    MSE_ConvNext = mean_squared_error(ConvNext['Actual_scores'], ConvNext['Predicted'])

    print(f'ResNet50: {MSE_ResNet}...\n')
    print(f'SimpleNet: {MSE_SimpleNet}...\n')
    print(f'Hyper: {MSE_Hyper}...\n')
    print(f'ConvNext: {MSE_ConvNext}...\n')

def calculate_RMSE():
    with open('ResNet50.pkl', 'rb') as handle:
        ResNet50 = pickle.load(handle)
    with open('SimpleNet.pkl', 'rb') as handle:
        SimpleNet = pickle.load(handle)
    with open('HyperNet.pkl', 'rb') as handle:
        Hyper = pickle.load(handle)
    with open('ConvNext.pkl', 'rb') as handle:
        ConvNext = pickle.load(handle)

    RMSE_ResNet = math.sqrt(mean_squared_error(ResNet50['Actual_scores'], ResNet50['Predicted']))
    RMSE_SimpleNet = math.sqrt(mean_squared_error(SimpleNet['Actual_scores'], SimpleNet['Predicted']))
    RMSE_Hyper = math.sqrt(mean_squared_error(Hyper['Actual_scores'], Hyper['Predicted']))
    RMSE_ConvNext = math.sqrt(mean_squared_error(ConvNext['Actual_scores'], ConvNext['Predicted']))

    print(f'ResNet50: {RMSE_ResNet}...\n')
    print(f'SimpleNet: {RMSE_SimpleNet}...\n')
    print(f'Hyper: {RMSE_Hyper}...\n')
    print(f'ConvNext: {RMSE_ConvNext}...\n')


def plt_MSE():
    with open('ResNet50.pkl', 'rb') as handle:
        ResNet50 = pickle.load(handle)
    with open('SimpleNet.pkl', 'rb') as handle:
        SimpleNet = pickle.load(handle)
    with open('HyperNet.pkl', 'rb') as handle:
        Hyper = pickle.load(handle)
    with open('ConvNext.pkl', 'rb') as handle:
        ConvNext = pickle.load(handle)

    MSE_ResNet = mean_squared_error(ResNet50['Actual_scores'], ResNet50['Predicted'])
    difference_array = np.subtract(np.array(ResNet50['Actual_scores']), np.array(ResNet50['Predicted']))
    SE_ResNet = np.square(difference_array)

    MSE_SimpleNet = mean_squared_error(SimpleNet['Actual_scores'], SimpleNet['Predicted'])
    difference_array = np.subtract(np.array(SimpleNet['Actual_scores']), np.array(SimpleNet['Predicted']))
    SE_SimpleNet = np.square(difference_array)

    MSE_Hyper = mean_squared_error(Hyper['Actual_scores'], Hyper['Predicted'])
    difference_array = np.subtract(np.array(Hyper['Actual_scores']), np.array(Hyper['Predicted']))
    SE_Hyper = np.square(difference_array)

    MSE_ConvNext = mean_squared_error(ConvNext['Actual_scores'], ConvNext['Predicted'])
    difference_array = np.subtract(np.array(ConvNext['Actual_scores']), np.array(ConvNext['Predicted']))
    SE_ConvNext = np.square(difference_array)


    # plt.plot(ResNet50['Actual_scores'], ResNet50['Predicted'], label='ResNet50')
    # plt.plot(SimpleNet['Actual_scores'], SimpleNet['Predicted'], label='SimpleNet')
    # plt.plot(Hyper['Actual_scores'], Hyper['Predicted'], label='Hyper Network')
    # plt.plot(ConvNext['Actual_scores'], ConvNext['Predicted'], label='ConvNext')
    plt.plot(SE_ResNet, label='ResNet50')
    plt.plot(SE_SimpleNet, label='SimpleNet')
    plt.plot(SE_Hyper, label='Hyper Network')
    plt.plot(SE_ConvNext, label='ConvNext')
    plt.ylabel('Square Error')
    plt.xlabel('Image sample')
    plt.legend()
    plt.show()
#===========================================#
def load_model(model_name):
    if model_name == 'ResNet50':
        model = load_ResNet50()
    elif model_name == 'SimpleNet':
        model = SimpleNet()
    elif model_name == 'HyperNet':
        import models_H
        model = models_H.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    elif model_name == 'ConvNext':
        model = load_convnext()

    model = model.cuda()

    model.load_state_dict(torch.load('Checkpoints/'+model_name+'_DBTex.pt'))
    model.eval()

    return model
#=============================================#
def run_1():
    # root = 'DATA/DBTex_Data/Test'
    # root = 'DATA/DBTex_2/Test'
    root = 'DATA/DBTex_Data_2_crop/Test'

    names, imgs, labels = get_data(root)


    model = load_model()

    actual_data = []
    predict_data = []
    with torch.no_grad():
        for item in range(imgs.shape[0]):
            img = imgs[item].cuda()
            label = labels[item]
            actual_data.append(label)
            pred = model(img)

            pred = torch.mean(pred)
            predict_data.append(pred.item() * 100)

            print(names[item])
    data_dic = {'Image_Name': names, 'Acutal': actual_data, 'Predicted': predict_data}
    df = pd.DataFrame(data_dic, columns=['Image_Name', 'Acutal', 'Predicted'])
    df.to_csv('_testOnly.csv', sep='\t', index=False, header=True)

    # for i in range(imgs.shape[0]):
    #     for j in range(25):
    #         plt.subplot(5,5,j+1)
    #         plt.imshow(imgs[i][j].permute(1,2,0))
    #     plt.show()
def run_2():
    # root = 'DATA/DBTex_Data/Test'
    # root = 'DATA/DBTex_2/Test'
    # root = 'DATA/DBTex_Data_2_crop/Test'
    root = 'DATA/DBTex_Data_3_crop_KeySlice/Test'
    names, imgs= get_data_test(root)


    # for i in range(imgs.shape[0]):
    #     for j in range(25):
    #         plt.subplot(5,5,j+1)
    #         plt.imshow(imgs[i][j].permute(1,2,0))
    #     plt.show()
    model_names=['ResNet50','SimpleNet','HyperNet','ConvNext']
    model_name = model_names[3]
    model = load_model(model_name)

    with open(os.path.join(root.split('/')[0], root.split('/')[1] + '/labels.pkl'), 'rb') as handle:
        labels = pickle.load(handle)
    # actual_data = []
    # for n in names:
    actual_data = []
    predict_data = []
    with torch.no_grad():
        for item in range(imgs.shape[0]):
            img = imgs[item].cuda()
            label = labels[names[item]].pop()
            actual_data.append(label)
            if model_name == 'HyperNet':
                import models_H
                # Generate weights for target network
                paras = model(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models_H.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
            else:
                pred = model(img)

            pred = torch.mean(pred)
            predict_data.append(pred.item())

            print(names[item])



    data_dic = {'Image_Name':names,'Actual_scores': actual_data, 'Predicted': predict_data}
    data_dic2 = {'Actual_scores': actual_data, 'Predicted': predict_data}
    with open(model_name+'.pkl', 'wb') as handle:
        pickle.dump(data_dic2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df = pd.DataFrame(data_dic)
    df.to_csv(model_name+'_testOnly.csv', sep='\t', index=False, header=True)

    print()
    # plot_scatter(actual_data,predict_data)

def check_key_slice():
    root_path = 'DATA/Key_Slice/'
    # root_path = 'DATA/DBTex_Data_3_crop_KeySlice/Test/03_Distorted_Images/'
    image_names, imgs = get_data_keys(root_path)
    # image_names, imgs = get_data_general(root_path)


    model_name = 'ConvNext'
    model = load_model(model_name)
    model.eval()
    predict_data = []
    with torch.no_grad():
        for item in range(imgs.shape[0]):
            img = imgs[item].cuda()
            if model_name == 'HyperNet':
                import models_H
                # Generate weights for target network
                paras = model(img)  # 'paras' contains the network weights conveyed to target network

                # Building target network
                model_target = models_H.TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                pred = model_target(paras['target_in_vec'])  # while 'paras['target_in_vec']' is the input to target net
            else:
                pred = model(img)

            pred = torch.mean(pred)
            # predict_data.append(pred.item())

            print(f'{image_names[item]} || QA: {pred}')

    # image_names = sorted(os.listdir(root_path))
    # for img_name in image_names:
    #     # img = pil_loader(os.path.join(root_path, img_name))
    #     view = img_name.split('_')[1].split('_')[0]
    #     img = Image.open(os.path.join(root_path, img_name))
    #     img = np.asarray(img)
    #     img = crop_Breast(img,view)
    #     print()

if __name__ == '__main__':
    # run_2()
    # plot_scatter_all()
    # plt_MSE()
    calculate_MSE()
    # calculate_RMSE()
    # check_key_slice()
