import torch
import visdom
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import DataSet
import torch.nn.functional as F


# 张量展平
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    # 将x展平为二维张量,且x的第一维大小保持不变
    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


# 可视化图片加载
def test():
    viz = visdom.Visdom()
    db = DataSet.Pokemon(r'D:\Datasets\10ClassesDataset', 224, 'train')
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=0)
    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))


# 准确率评估
def evaluate(model, loader):
    # 为0到9十个label都创建一个空列表,以label:空列表[]的形式形成一个字典all_preds
    all_preds = {label: [] for label in range(len(loader.dataset.name2label))}
    all_preds_prob = {label: [] for label in range(len(loader.dataset.name2label))}
    model.eval()
    all_logits = []
    all_labels = []

    correct = 0
    total = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(torch.device('cuda')), y.to(torch.device('cuda'))
        with torch.no_grad():
            # logits = F.softmax(model(x), dim=1)
            logits = model(x)
            preds_prob, preds = torch.max(logits, 1)  # 收集每次预测的概率和类别,用于绘制箱型图
            # print(preds_prob)
            # print(preds)
            all_logits.append(logits)
            all_labels.append(y)
            for p, prob, label in zip(preds.cpu().numpy(), preds_prob.cpu().numpy(), y.cpu().numpy()):
                all_preds[label].append(p)
                all_preds_prob[label].append(prob)
        correct += torch.eq(preds, y).sum().float().item()
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return correct / total, all_logits.cpu().numpy(), all_labels.cpu().numpy(), all_preds, all_preds_prob


# 画ROC曲线
def plot_roc_curve(all_logits, all_labels, n_classes=10):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %i' % (roc_auc[i], i))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# 绘制箱型图
def plot_boxplots(all_preds):
    # 将每个类别的值(即字典中key对应的一组values)转换为为numpy数组,然后这十个类别的numpy数组再一起组成pred_arrays
    pred_arrays = [torch.tensor(preds) for preds in all_preds.values()]
    # 绘制箱型图
    plt.figure(figsize=(10, 5))
    plt.boxplot(pred_arrays, labels=list(all_preds.keys()))
    plt.title('Class Predictions Boxplot')
    plt.xlabel('Class')
    plt.ylabel('Prediction Value/Probability')
    plt.show()
