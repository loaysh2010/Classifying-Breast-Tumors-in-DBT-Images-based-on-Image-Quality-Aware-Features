import os
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from Networks_classification import *
from Networks_IQA import load_model
# from torchvision import models
from prepair_data import load_split_train_test
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
#-------------------Load Data--------------------------#
data_dir = 'Data/01_Used_Data_1/'
class_names = ['Benign','Malignant']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
dataset = datasets.ImageFolder(data_dir, transform=transform)
targets=dataset.targets

kfold = StratifiedKFold(n_splits=5,shuffle=True)
acc_scores=[]
pr_scores =[]
re_scores = []
f1_scores=[]
#============================================================#
k=0
for train_index, test_index in kfold.split(dataset,targets):

    train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_index)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=8,sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=8,sampler=test_sampler)
    #=============================================++#
    # ==========================================================#
    # -------------------Load Model--------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # freez = False
    # model_name, model = load_AlexNet(freez=freez)
    # save_name = model_name + '_fineTune'

    model_name = 'Mymodel_Alex'
    save_name = model_name + '_DBT'
    model = Mymodel()

    model.cuda()
    print(f'Training ==> {save_name}')
    print(60 * '-')

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    epochs = 50
    k+=1
    best_train_acc = 0.0
    best_test_acc = 0.0
    for epoch in range(epochs):
        print(f"fold:{k} ==> Epoch {epoch + 1}/{epochs}.. ")
        print(10 * '-')

        model.train()
        tr_correct, tr_total = 0, 0
        y_pred = []
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model(inputs)
            _, pred = torch.max(logps.data, 1)

            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        # --------- Validation-------------#
        test_running_loss = 0
        test_correct, test_total = 0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(testloader):
                inputs, labels = inputs.to(device), labels.to(device)

                logps = model(inputs)
                _, pred = torch.max(logps.data, 1)

                loss = criterion(logps, labels)
                test_running_loss += loss.item()

                test_total += labels.size(0)
                test_correct += (pred == labels).sum().item()

            test_loss = test_running_loss / len(testloader)

            test_acc = test_correct / test_total
            if best_test_acc < test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'checkpoints2/' + save_name + '_k'+str(k)+'.pth')
                print('Model Saved...')
            print(60 * '-')

        print(f"Test loss: {test_loss:.3f}.. "
              f"Test Acc : {test_acc * 100:.3f}.. ")

    model.load_state_dict(torch.load('checkpoints2/' + save_name + '_k'+str(k)+'.pth', map_location=torch.device('cpu')))
    model = model.to(device)
    print(f'Testing ==> {model_name}')
    print(60 * '-')
    model.eval()
    test_label = []
    test_pred = []
    test_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            test_label += labels.cpu()

            logps = model(inputs)
            _, pred = torch.max(logps.data, 1)
            test_pred += pred.cpu()
            loss = criterion(logps, labels)
            test_loss += loss.item()

            total += labels.size(0)
            correct += (pred == labels).sum().item()

        test_loss = test_loss / len(testloader)
        test_acc = correct / total

    print(60 * '-')
    classes = testloader.dataset.classes
    test_label = np.asarray(test_label)
    y_pred = np.asarray(test_pred)
    confusion_matrix = np.zeros((2, 2))
    for t, p in zip(test_label, y_pred):
        confusion_matrix[t, p] += 1

    tp = confusion_matrix[1,1]
    tn = confusion_matrix[0,0]
    fn = confusion_matrix[1,0]
    fp = confusion_matrix[0,1]
    acc = (tp+tn)/(tp+tn+fn+fp)
    pre = (tp)/(tp+fp)
    rec = (tp)/(tp+fn)
    f1  = 2*pre*rec/(pre + rec)
    print(f"Fold:{k} => acc:{acc}, pre:{pre},rec:{rec},f1-score={f1}")
    acc_scores.append(acc)
    pr_scores.append(pre)
    re_scores.append(rec)
    f1_scores.append(f1)

print(acc_scores)
print(pr_scores)
print(re_scores)
print(f1_scores)
