import torch
import torch.nn as nn

import torchvision.models as models
# from uniformer import uniformer_base,uniformer_small,uniformer_base_ls,uniformer_small_plus

class Model(nn.Module):
    def __init__(self,feature):
        super(Model, self).__init__()
        self.Feature = feature
        

    def forward(self, x):
        x = self.Feature.forward_features(x) # For uniformer
        # x = self.Feature(x)
        return torch.flatten(x,start_dim=1)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.Layer_1 = nn.Sequential(
                            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2))

        self.Layer_2 = nn.Sequential(
                            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2))

        self.Layer_3 = nn.Sequential(
                            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2))

        self.Layer_4 = nn.Sequential(
                            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2))

        self.Layer_5 = nn.Sequential(
                            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2))

        self.linear_layers = nn.Sequential(
            nn.Linear(25088, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.Layer_1(x)
        x = self.Layer_2(x)
        x = self.Layer_3(x)
        x = self.Layer_4(x)
        x = self.Layer_5(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def load_ResNet50():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features,out_features=1,bias=True)

    return model

def load_ResNet101():
    resnet101 = models.resnet101(pretrained=True)
    resnet101 = torch.nn.Sequential(*(list(resnet101.children())[:-1]))
    model = Model(resnet101)
    # print(model)
    return model

def load_convnext():
    model = models.convnext_tiny(pretrained=True)
    model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features,out_features=1,bias=True)
    return model


def load_DensNet():
    DensNet = models.densenet161(pretrained=True)
    DensNet = torch.nn.Sequential(*(list(DensNet.children())[:-1]))

    model = Model(DensNet)
    return model

def load_model(model_name,device='cuda'):
    if model_name == 'ResNet50':
        model = load_ResNet50()
    elif model_name == 'SimpleNet':
        model = SimpleNet()
    elif model_name == 'HyperNet':
        import models_H
        model = models_H.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    elif model_name == 'ConvNext':
        model = load_convnext()

    # model = model.to(device)

    model.load_state_dict(torch.load('checkpoints/'+model_name+'_DBTex.pt',map_location=torch.device('cpu')))
    # model.eval()

    return model

if __name__ == '__main__':
    # model = load_ResNet101()
    # model = load_DensNet()
    # model = load_Uniformer()
    # model = SimpleNet()
    model = load_convnext()
    inp = torch.rand((1,3,32,32))
    out = model(inp)
    print(model)