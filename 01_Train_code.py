from torch import optim
from tqdm import tqdm
from Networks_classification import *
from prepair_data import load_split_train_test
#===========================================================#
#-------------------Load Data--------------------------#
data_dir = 'Data/01_Used_Data_2/01_Origin_Data/'
# data_dir = 'Data/02_Inbreast/01_Origin_Data/'

trainloader, testloader = load_split_train_test(data_dir, 8)
print(60*'-')
# print(f'Dataset classes: {trainloader.dataset.classes}')
print(f'Number of Train Data: {len(trainloader.dataset)}')
print(f'Number of Test Data: {len(testloader.dataset)}')
print(60*'-')
#===============================================================
#-------------------Load Model--------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
#======================================================#
model.to(device)
save_name = model_name + '_DBT'
# save_name = model_name + '_Inbreast'

print(f'Training ==> {save_name}')
print(60*'-')
#================================================#
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#=======================================================#
epochs = 50
best_train_acc = 0.0
best_test_acc = 0.0
tr_losses=[]
test_losses=[]
tr_Accuracy=[]
test_Accuracy=[]
for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}.. ")
        print(10*'-')

        model.train()
        tr_running_loss = 0
        running_acc = 0
        tr_correct,tr_total=0,0
        y_pred = []
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model(inputs)
            _, pred = torch.max(logps.data, 1)

            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            tr_running_loss += loss.item()

            tr_total += labels.size(0)
            tr_correct += (pred == labels).sum().item()
        train_loss = tr_running_loss / len(trainloader)
        tr_losses.append(train_loss)

        train_acc = tr_correct / tr_total
        tr_Accuracy.append(train_acc)
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
            test_losses.append(test_loss)

            test_acc = test_correct / test_total
            test_Accuracy.append(test_acc)

        print(60 * '-')
        print(f"Train loss: {train_loss:.3f}.. "
              f"Train accuracy: {train_acc * 100:.3f}.. ")

        print(f"Test loss: {test_loss:.3f}.. "
              f"Test accuracy: {test_acc * 100:.3f}.. ")
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/'+save_name +'.pth')
            print('Model Saved...')
