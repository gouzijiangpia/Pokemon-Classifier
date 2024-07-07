import time
import torch
import visdom
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights, vgg11, VGG11_Weights
import DataSet
from utils import showprocess, Flatten, plot_roc_curve, evaluate, plot_boxplots
from Vit import ViT

# 评估模型准确率

if __name__ == '__main__':
    '''init'''
    start = time.time()
    viz = visdom.Visdom()
    showprocess()
    batchsize = 32
    epochs = 15
    device = torch.device('cuda')
    torch.manual_seed(1234)
    Pokemon = DataSet.Pokemon
    modelflag = 2
    '''自定义数据集'''
    # 获取数据集datasets
    train_db = Pokemon(r'D:\Datasets\10ClassesDataset', 224, mode='train')
    val_db = Pokemon(r'D:\Datasets\10ClassesDataset', 224, mode='val')
    test_db = Pokemon(r'D:\Datasets\10ClassesDataset', 224, mode='test')
    # 用datasets创建loader对象
    train_loader = DataLoader(train_db, batch_size=batchsize, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_db, batch_size=batchsize, num_workers=0)
    test_loader = DataLoader(test_db, batch_size=batchsize, num_workers=0)

    if modelflag == 0:  # 创建Vision Transformer
        vit = ViT().to(device)
        model = vit
    else:  # 创建resnet18、vgg11
        if modelflag == 1:
            # trained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
            trained_model = resnet18(weights=None)
            model = nn.Sequential(*list(trained_model.children())[:-1],
                                  Flatten(),
                                  nn.Linear(512, 10)
                                  ).to(device)
        elif modelflag == 2:
            trained_model = vgg11(weights=None)
            model = nn.Sequential(*list(trained_model.children())[:-1],  # [b, 512, 1, 1]
                                  Flatten(),  # [b, 512, 1, 1] => [b, 512]
                                  nn.Linear(25088, 10)
                                  ).to(device)
    # x = torch.randn(2, 3, 224, 224).to(device)
    x = torch.randn(2, 3, 64, 64).to(device)
    # print(model)
    # print(model(x).shape)

    '''train'''
    optimizer = optim.Adam(model.parameters(), lr=0.0008)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_epoch = 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss', xlabel='epoch', ylabel='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc', xlabel='global_step', ylabel='acc'))

    for epoch in range(epochs):
        # 训练
        for step, (x, y) in enumerate(train_loader):
            # x: [b, 3, 224, 224], y: [b]
            x, y = x.to(device), y.to(device)

            model.train()
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            # 每训练一批数据global_step+1
            global_step += 1

        # 验证
        if epoch % 1 == 0:
            val_acc, _, _, _, _ = evaluate(model, val_loader)
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                if modelflag == 0:
                    torch.save(model.state_dict(), 'best0.mdl')
                else:
                    if modelflag == 1:
                        torch.save(model.state_dict(), 'best1_1.mdl')
                    elif modelflag == 2:
                        torch.save(model.state_dict(), 'best2_2.mdl')

            viz.line([val_acc], [global_step], win='val_acc', update='append')
            print(f'epoch: {epoch}, val_acc: {val_acc}')

    print('best acc:', best_acc, 'best epoch:', best_epoch)
    if modelflag == 0:
        model.load_state_dict(torch.load('best0.mdl'))
    else:
        if modelflag == 1:
            model.load_state_dict(torch.load('best1_1.mdl'))
        elif modelflag == 2:
            model.load_state_dict(torch.load('best2_2.mdl'))
    print('loaded from ckpt!')
    # 测试
    test_acc, all_logits, all_labels, all_preds, all_preds_prob = evaluate(model, test_loader)
    print('test acc:', test_acc)
    print('time:', time.time() - start)
    # 根据测试结果绘制ROC曲线和箱型图
    plot_roc_curve(all_logits, all_labels, 10)
    plot_boxplots(all_preds)
    # plot_boxplots(all_preds_prob)
