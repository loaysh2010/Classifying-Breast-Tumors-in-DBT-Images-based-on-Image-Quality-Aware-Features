import torch
import torch.nn as nn
from torchvision import models
from Networks_IQA import load_model
from timm.models.ghostnet import ghostnet_050
from timm.models.hrnet import hrnet_w18
from timm.models.coat import coat_tiny
from timm.models.cspnet import darknet53
#=========================================#
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        self.QA_model = load_model('ConvNext')
        self.QA_model.classifier = nn.Identity()
        # self.QA_model = self.QA_model.features.append(self.QA_model.avgpool)
        for param in self.QA_model.parameters():
            param.requires_grad = False

        model_name,self.MC = load_AlexNet(False)
        # model_name,self.MC = load_Vgg16(False)
        # model_name,self.MC = load_ResNet50(False)
        # model_name,self.MC = load_DensNet121(False)
        # model_name,self.MC = load_efficientNetb0(False)
        # model_name,self.MC = load_DarkNet53(False)
        # model_name,self.MC = load_GhostNet_50(False)
        # model_name,self.MC = load_HRNet_18(False)
        # model_name,self.MC = load_CoAtNet(False)
        # model_name,self.MC = load_convNext(False)
        # save_name = model_name + '_fineTune'
        # self.MC.load_state_dict(torch.load('checkpoints/' + save_name + '_[from_Test].pth'))
        # for param in self.MC.parameters():
        #     param.requires_grad = False
        # self.MC = models.alexnet(True)
        # self.MC = models.vgg16(True)
        # self.MC = models.resnet50(True)
        # self.MC = models.densenet121(pretrained=True)
        # self.MC = models.efficientnet_b0(pretrained=True)
        # self.MC = models.convnext_base(pretrained=True)
        # self.MC = models.convnext_tiny(pretrained=True)
        # self.MC = ghostnet_050(pretrained=False)
        # self.MC = hrnet_w18(pretrained=True)
        # self.MC = coat_tiny(pretrained=True)
        # self.MC = darknet53(pretrained=True)
        # for param in self.MC.parameters():
        #     param.requires_grad = False

        self.MC.classifier = nn.Identity()
        # self.MC.classifier = nn.Identity()
        # self.MC.fc = nn.Identity()
        # self.MC.head = nn.Identity()
        # self.MC.head.fc = nn.Identity()


        self.flatten = nn.Flatten()
        self.middel = nn.Linear(9216, 768, bias=True)
        # self.middel = nn.Linear(25088, 768, bias=True)
        # self.middel = nn.Linear(2048, 768, bias=True)
        # self.middel = nn.Linear(1024, 768, bias=True)
        # self.middel = nn.Linear(1280, 768, bias=True)
        # self.middel = nn.Linear(152, 768, bias=True)

        self.classifier = nn.Linear(768, 2, bias=True)
        # self.classifier = nn.Linear(1536,2,bias=True)
        # self.classifier = nn.Linear(1792,2,bias=True)

        self.gradients = None


    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        a = self.QA_model(x)
        # register the hook
        # h = x.register_hook(self.activations_hook)
        a= self.flatten(a)
        b = self.MC(x)
        # h = x.register_hook(self.activations_hook)
        b = self.flatten(b)
        m = self.middel(b)
        o = a + m
        # o = (0.4 * a) + (0.6*m)
        # o = torch.cat([a,m],1)
        o = self.classifier(o)
        return o

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.MC(x)

def load_AlexNet(freez = True):
    model_name = 'AlexNet'
    model = models.alexnet(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features,out_features=2, bias=True)
        for param in model.classifier[6].parameters():
            param.requires_grad = True
    else:
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features,out_features=2, bias=True)

    return model_name, model

def load_Vgg16(freez = True):
    model_name = 'Vgg16'
    model = models.vgg16(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=2, bias=True)
        for param in model.classifier[6].parameters():
            param.requires_grad = True
    else:
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=2, bias=True)

    return model_name, model
def load_Vgg19(freez = True):
    model_name = 'Vgg19'
    model = models.vgg19(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=2, bias=True)
        for param in model.classifier[6].parameters():
            param.requires_grad = True
    else:
        model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=2, bias=True)

    return model_name, model

def load_ResNet50(freez = True):
    model_name = 'ResNet50'
    model = models.resnet50(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)

    return model_name, model
def load_ResNet101(freez = True):
    model_name = 'ResNet101'
    model = models.resnet101(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)

    return model_name, model

def load_SqueezNet(freez = True):
    model_name = 'SqueezNet'
    model = models.squeezenet1_1(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Conv2d(in_channels=model.classifier[1].in_channels, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        model.classifier[1] = nn.Conv2d(in_channels=model.classifier[1].in_channels, out_channels=2, kernel_size=(1, 1), stride=(1, 1))

    return model_name, model

def load_DensNet121(freez = True):
    model_name = 'DensNet121'
    model = models.densenet121(pretrained=False)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=2, bias=True)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=2, bias=True)
    return model_name, model
def load_DensNet201(freez = True):
    model_name = 'DensNet201'
    model = models.densenet201(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=2, bias=True)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=2, bias=True)
    return model_name, model

def load_Inception(freez = True):
    model_name = 'InceptionV3'
    model = models.inception_v3(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, out_features=2, bias=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
        # model.fc.out_features=2
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, out_features=2, bias=True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)

    return model_name, model
def load_GoogleNet(freez = True):
    model_name = 'GoogleNet'
    model = models.googlenet(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)

    return model_name, model
def load_ShuffleNet(freez=True):
    model_name = 'ShuffleNet'
    model = models.shufflenet_v2_x0_5(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)

    return model_name, model
def load_MobileNet(freez=True):
    model_name = 'MobileNetv2'
    model = models.mobilenet_v2(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2, bias=True)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2, bias=True)

    return model_name, model

def load_efficientNetb0(freez=True):
    model_name = 'efficientNetB0'
    model = models.efficientnet_b0(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2, bias=True)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2, bias=True)

    return model_name, model
def load_efficientNetb7(freez=True):
    model_name = 'efficientNetB7'
    model = models.efficientnet_b7(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2, bias=True)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2, bias=True)

    return model_name, model

def load_wideResNet50(freez=True):
    model_name = 'wideResNet50'
    model = models.wide_resnet50_2(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)

    return model_name, model

    return model_name, model
def load_wideResNet101(freez=True):
    model_name = 'wideResNet101'
    model = models.wide_resnet101_2(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=2, bias=True)

    return model_name, model

def load_ViT(freez=True):
    model_name = 'ViT_Base'
    model = models.vit_b_32(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=2, bias=True)
        for param in model.heads.head.parameters():
            param.requires_grad = True
    else:
        model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=2, bias=True)

    return model_name, model
def load_convNext(freez=True):
    model_name = 'ConvNext_Base'
    model = models.convnext_base(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features, out_features=2, bias=True)
        for param in model.classifier[2].parameters():
            param.requires_grad = True
    else:
        model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features, out_features=2, bias=True)

    return model_name, model


def load_GhostNet_50(freez=True):
    model_name = 'GhostNet_50'
    model = ghostnet_050(pretrained=False)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=2, bias=True)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=2, bias=True)

    return model_name, model
def load_HRNet_18(freez=True):
    model_name = 'HRNet_18'
    model = hrnet_w18(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=2, bias=True)
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=2, bias=True)

    return model_name, model
def load_CoAtNet(freez=True):
    model_name = 'coAtNet'
    model = coat_tiny(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.head = nn.Linear(in_features=model.head.in_features, out_features=2, bias=True)
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        model.head = nn.Linear(in_features=model.head.in_features, out_features=2, bias=True)

    return model_name, model
def load_DarkNet53(freez=True):
    model_name = 'darkNet53'
    model = darknet53(pretrained=True)
    if freez:
        for param in model.parameters():
            param.requires_grad = False
        model.head.fc  = nn.Linear(in_features=model.head.fc.in_features, out_features=2, bias=True)
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        model.head.fc = nn.Linear(in_features=model.head.fc.in_features, out_features=2, bias=True)

    return model_name, model

