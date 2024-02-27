import numpy as np
from tqdm import tqdm
from Networks_classification import *
from prepair_data import load_split_train_test
#===========================================================#
#-------------------Load Data--------------------------#
data_dir = 'Data/01_Used_Data_2/01_Origin_Data/'
# data_dir = 'Data/02_Inbreast/01_Origin_Data/'

_, testloader = load_split_train_test(data_dir, 8)
print(60*'-')
# print(f'Dataset classes: {trainloader.dataset.classes}')
# print(f'Number of Train Data: {len(trainloader.dataset)}')
print(f'Number of Test Data: {len(testloader.dataset)}')
num_classes =len(testloader.dataset.classes)
print(60*'-')
#===============================================================
#-------------------Load Model--------------------------#
#-------------------Load Model--------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
freez = False

#========= Test Baseline models ====================#
# model_name,model = load_AlexNet(freez = freez)
# model_name,model = load_Vgg16(freez = freez)
# model_name,model = load_Vgg19(freez = freez)
# model_name,model = load_ResNet50(freez = freez)
# model_name,model = load_ResNet101(freez = freez)
# model_name,model = load_SqueezNet(freez = freez)
# model_name,model = load_DensNet201(freez = freez)
# model_name,model = load_Inception(freez = freez)
# model_name,model = load_GoogleNet(freez = freez)
# model_name,model = load_ShuffleNet(freez = freez)
# model_name,model = load_MobileNet(freez = freez)
# model_name,model = load_efficientNetb0(freez = freez)
# model_name,model = load_efficientNetb7(freez = freez)
# model_name,model = load_wideResNet50(freez = freez)
# model_name,model = load_wideResNet101(freez = freez)
# model_name,model = load_ViT(freez=freez)
# model_name,model = load_convNext(freez=freez)
# save_name = model_name + '_best'

#========= Test Proposed models ====================#
model_name = 'Mymodel_Alex'
# model_name = 'Mymodel_VGG16'
# model_name = 'Mymodel_ResNet50'
# model_name = 'Mymodel_DensNet'
# model_name = 'Mymodel_efficientNet2'
# model_name = 'Mymodel_ResNet101'
# model_name = 'Mymodel_ConvNext2'
# model_name = 'Mymodel_GhostNet502'
# model_name = 'Mymodel_HRNet502'
# model_name = 'Mymodel_CoAtNet2'
# model_name = 'Mymodel_DarkNet532'

model = Mymodel()
save_name = model_name + '_DBT'
#============================================================#

model.load_state_dict(torch.load('checkpoints/'+save_name +'.pth',map_location=torch.device('cpu')))
model = model.to(device)
print(f'Testing ==> {model_name}')
print(60*'-')
#====================================================================++#
#-------------------Test DL Model --------------------------#
criterion = nn.CrossEntropyLoss()

model.eval()
test_label = []
test_pred = []
test_loss = 0
correct,total=0,0
with torch.no_grad():
        for inputs, labels in tqdm(testloader):
                inputs, labels = inputs.to(device), labels.to(device)
                test_label += labels.cpu()

                logps = model(inputs)
                _, pred = torch.max(logps.data, 1)
                test_pred += pred.cpu()
                loss = criterion(logps, labels)
                test_loss += loss.item()

                total+=labels.size(0)
                correct+=(pred==labels).sum().item()

        test_loss = test_loss / len(testloader)
        test_acc = correct/total

print(f"Test loss: {test_loss:.3f}.. "
      f"Test Acc : {test_acc*100:.3f}.. ")
#================================================#
# Build confusion matrix
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd

classes = testloader.dataset.classes
test_label = np.asarray(test_label)
y_pred = np.asarray(test_pred)
confusion_matrix = np.zeros((num_classes, num_classes))
for t, p in zip(test_label, y_pred):
    confusion_matrix[t, p] += 1

plt.figure(figsize=(15,10))

class_names = list(classes)
df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names).astype(int)
heatmap = sn.heatmap(df_cm, annot=True,cmap="Blues", fmt="d") # cmap = ["viridis","coolwarm","cividis","crest"]
plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.savefig(dataset+'/plots/'+model_name+'_Confusion_plot.png')
plt.show()