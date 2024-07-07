import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights, vgg11, VGG11_Weights
from torch import nn
from utils import Flatten, evaluate, plot_roc_curve, plot_boxplots
import DataSet
from Vit import ViT

modelflag = 2
modelname = 'Vit'
if modelflag == 0:
    model = ViT().to(torch.device('cuda'))
    model.load_state_dict(torch.load('best0.mdl'))
else:
    if modelflag == 1:
        modelname = 'ResNet18'
        trained_model = resnet18(weights=None)
        model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                              Flatten(),  # [b, 512, 1, 1] => [b, 512]
                              nn.Linear(512, 10)
                              ).to(torch.device('cuda'))
        model.load_state_dict(torch.load('best1_1.mdl'))
    elif modelflag == 2:
        modelname = 'VGG11'
        trained_model = vgg11(weights=None)
        model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                              Flatten(),  # [b, 512, 1, 1] => [b, 512]
                              nn.Linear(25088, 10)
                              ).to(torch.device('cuda'))
        model.load_state_dict(torch.load('best2_2.mdl'))
model.eval()
test_db = DataSet.Pokemon(r'D:\Datasets\10ClassesDataset', 224, mode='test')
test_loader = DataLoader(test_db, batch_size=32, num_workers=0)
test_acc, all_logits, all_labels, all_preds, all_preds_prob = evaluate(model, test_loader)
print('model:', modelname, ' test acc:', test_acc)
plot_roc_curve(all_logits, all_labels, 10)
plot_boxplots(all_preds)
# plot_boxplots(all_preds_prob)
